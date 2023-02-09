#include "MRSharpenMarchingCubesMesh.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRTriMath.h"
#include "MRTimer.h"

namespace MR
{

void sharpenMarchingCubesMesh( const Mesh & ref, Mesh & vox, Vector<VoxelId, FaceId> & face2voxel,
    const SharpenMarchingCubesMeshSettings & settings )
{
    MR_TIMER
    assert( settings.minNewVertDev < settings.maxNewVertDev );
    Vector<Vector3f, VertId> normals( vox.topology.vertSize() );
    // find normals and correct points
    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, normals.endId() ), [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( !vox.topology.hasVert( v ) )
                continue;
            const auto proj = findProjection( vox.points[v], ref );

            Vector3f n = ( vox.points[v] - proj.proj.point ).normalized();
            Vector3f np = ref.pseudonormal( proj.mtp );
            if ( settings.offset == 0 || n.lengthSq() <= 0 )
                n = np;
            else if ( dot( n, np ) < 0 )
                n = -n;

            if ( settings.maxOldVertPosCorrection > 0 )
            {
                const auto newPos = proj.proj.point + settings.offset * n;
                if ( ( newPos - vox.points[v] ).lengthSq() <= sqr( settings.maxOldVertPosCorrection ) )
                    vox.points[v] = newPos;
                else
                    n = Vector3f{}; //undefined
            }

            normals[v] = n;
        }
    } );

    auto facesToProcess = vox.topology.getValidFaces();
    VertId firstNewVert( vox.topology.vertSize() );
    // line directions in new vertices, dirs[i] contains the data for vertex firstNewVert+i
    std::vector<Vector3f> dirs;
    for ( auto f : facesToProcess )
    {
        const auto voxel = face2voxel[f];
        EdgeId e0 = vox.topology.edgeWithLeft( f );
        EdgeId e = e0;
        for (;;)
        {
            auto r = vox.topology.right( e );
            if ( !r || face2voxel[r] != voxel )
                break;
            e = vox.topology.prev( e );
            if ( e == e0 )
            {
                assert( false ); //not found after full cycle
                break;
            }
        }
        e0 = e; // an edge with this voxel on the left and another voxel on the right
        Vector3f sumAC;
        float sumArea = 0;
        PlaneAccumulator pacc;
        do 
        {
            auto v = vox.topology.org( e );
            pacc.addPlane( Plane3f::fromDirAndPt( normals[v], vox.points[v] ) );

            auto ei = e;
            for ( ;; )
            {
                auto l = vox.topology.left( ei );
                if ( !l || face2voxel[l] != voxel )
                    break;
                if ( facesToProcess.test_set( l, false ) )
                {
                    auto a = vox.dblArea( l );
                    sumArea += a;
                    sumAC += a * vox.triCenter( l );
                }
                ei = vox.topology.next( ei );
            }

            e = vox.topology.prev( e.sym() );
            for (;;)
            {
                auto r = vox.topology.right( e );
                if ( !r || face2voxel[r] != voxel )
                    break;
                e = vox.topology.prev( e );
            }
        } while ( e != e0 );

        if ( sumArea <= 0 )
            continue; //degenerate triangles within voxel

        Vector3f avgPt = sumAC / sumArea;
        int rank = 0;
        Vector3f dir;
        constexpr float tol = 0.01f; // tolerance for comparing eigenvalues
        auto sharpPt = pacc.findBestCrossPoint( avgPt, tol, &rank, &dir );
        if ( rank <= 1 )
            continue; // the surface is planar within the voxel
        const auto distSq = ( avgPt - sharpPt ).lengthSq();
        if ( distSq < sqr( settings.minNewVertDev ) )
            continue; //too little deviation of new point to introduce a vertex in mesh
        if ( distSq > sqr( settings.maxNewVertDev ) )
            continue; //new point is too from existing mesh triangles

        auto v = vox.splitFace( f );
        assert( v == dirs.size() + firstNewVert );
        dirs.push_back( dir );
        vox.points.autoResizeSet( v, sharpPt );
        for ( auto ei : orgRing( vox.topology, v ) )
            face2voxel.autoResizeSet( vox.topology.left( ei ), voxel );

        // connect new vertex with every vertex from the voxel
        vox.topology.flipEdgesIn( v, [&]( EdgeId e )
        {
            auto r = vox.topology.right( e );
            assert( r );
            if ( face2voxel[r] == voxel )
            {
                [[maybe_unused]] auto b = vox.topology.dest( vox.topology.prev( e ) );
                assert( !vox.topology.findEdge( v, b ) ); //there is no edge between v and b yet
                return true;
            }
            return false;
        } );

        // make triangles from old voxel vertices if all 3 vertices have similar normals;
        // this reduces self-intersections appeared after previous flip
        vox.topology.flipEdgesOut( v, [&]( EdgeId e )
        {
            assert( vox.topology.org( e ) == v );
            auto b = vox.topology.dest( vox.topology.prev( e ) );
            auto c = vox.topology.dest( e );
            auto d = vox.topology.dest( vox.topology.next( e ) );
            SymMatrix3f mat;
            mat += outerSquare( normals[b] );
            mat += outerSquare( normals[c] );
            mat += outerSquare( normals[d] );
            const auto eigenvalues = mat.eigens();
            if ( eigenvalues[1] > eigenvalues[2] * tol )
                return false; // normals in the vertices are not equal for given tolerance
            if ( vox.topology.findEdge( d, b ) )
                return false; // multiple edges between b and d will appear
            return true;
        } );
    }

    // find edges between voxels with new vertices
    UndirectedEdgeBitSet sharpEdges( vox.topology.undirectedEdgeSize() );
    for ( auto v = firstNewVert; v < vox.topology.vertSize(); ++v )
    {
        const auto vDir = dirs[ v - firstNewVert ];
        const bool vIsCorner = vDir.lengthSq() < 0.1f; // unit length for edges
        // not-corner vertices can have at most two sharp edges with other new vertices
        struct CanditeEdge
        {
            float metric = 0;
            UndirectedEdgeId edge;
        };
        CanditeEdge best, secondBest; // first maximal metrics
        for ( auto ei : orgRing( vox.topology, v ) )
        {
            if ( !vox.topology.left( ei ) )
                continue;
            EdgeId e = vox.topology.prev( ei.sym() );
            if ( !vox.topology.right( e ) )
                continue;
            auto b = vox.topology.dest( vox.topology.prev( e ) );
            if ( b >= firstNewVert )
            {
                auto ap = vox.points[ vox.topology.org( e ) ];
                auto bp = vox.points[ b ];
                auto cp = vox.points[ vox.topology.dest( e ) ];
                auto dp = vox.points[ v ];
                auto nABD = normal( ap, bp, dp );
                auto nBCD = normal( bp, cp, dp );
                // allow creation of very sharp edges (like in default prism or in cone with 6 facets),
                // which isUnfoldQuadrangleConvex here did not allow;
                // but disallow making extremely sharp edges, where two triangle almost coincide with opposite normals
                if ( dot( nABD, nBCD ) >= settings.minNormalDot )
                {
                    if ( vIsCorner )
                        sharpEdges.set( e.undirected() );
                    else
                    {
                        const auto bDir = dirs[ b - firstNewVert ];
                        const bool bIsCorner = bDir.lengthSq() < 0.1f; // unit length for edges
                        const auto bvDir = ( vox.points[b] - vox.points[v] ).normalized();
                        // dot( vDir, bDir ) worked bad for cone vertex
                        const auto metric = bIsCorner ? 10.0f : std::abs( dot( vDir, bvDir ) );
                        if ( metric > 0.5f ) // avoid connection with vertex not along v-line
                        {
                            CanditeEdge c{ .metric = metric, .edge = e.undirected() };
                            if ( c.metric > best.metric )
                            {
                                secondBest = best;
                                best = c;
                            }
                            else if ( c.metric > secondBest.metric )
                            {
                                secondBest = c;
                            }
                        }
                    }
                }
            }
        }
        if ( best.edge )
            sharpEdges.set( best.edge );
        if ( secondBest.edge )
            sharpEdges.set( secondBest.edge );
    }

    // flip edges between voxels with new vertices to form sharp ridges
    for ( EdgeId e : sharpEdges )
    {
        auto b = vox.topology.dest( vox.topology.prev( e ) );
        auto d = vox.topology.dest( vox.topology.next( e ) );
        if ( !vox.topology.findEdge( b, d ) )
            vox.topology.flipEdge( e );
    }

    // best position new vertices on found lines
    std::vector<Vector3f> newPos;
    newPos.reserve( vox.topology.vertSize() - firstNewVert );
    for ( int iPosSel = 0; iPosSel < settings.posSelIters; ++iPosSel )
    {
        // calculate optimal position of each vertex independently
        newPos.clear();
        for ( auto iv = firstNewVert; iv < vox.topology.vertSize(); ++iv )
        {
            const auto p = vox.points[iv];
            newPos.push_back( p );
            const auto vDir = dirs[ iv - firstNewVert ];
            const bool vIsCorner = vDir.lengthSq() < 0.1f; // unit length for edges
            if ( vIsCorner )
                continue;
            float uv = 0, vv = 0;
            for ( auto ei : orgRing( vox.topology, iv ) )
            {
                if ( !vox.topology.left( ei ) )
                    continue;
                auto ap = vox.destPnt( ei );
                auto bp = vox.destPnt( vox.topology.next( ei ) );
                auto u = cross( bp - ap, p - ap );
                auto v = cross( bp - ap, vDir );
                uv += dot( u, v );
                vv += dot( v, v );
            }
            if ( vv > 0 )
                newPos.back() -= uv / vv * vDir;
        }

        // move each vertex half way toward its optimal position
        for ( auto v = firstNewVert; v < vox.topology.vertSize(); ++v )
        {
            const auto pOld = vox.points[v];
            const auto pNew = newPos[ v - firstNewVert ];
            vox.points[v] = 0.5f * ( pOld + pNew );
        }
    }

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );
}

} //namespace MR
