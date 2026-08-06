#include "MRSharpenMarchingCubesMesh.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRReducePath.h"
#include "MRLineSegm.h"
#include "MRVolumeIndexer.h"
#include "MRBox.h"
#include "MRMeshIntersect.h"

namespace MR
{

namespace
{
/// returns the two lattice nodes bounding the lattice edge, on which the vertex (v) of marching cubes mesh is located;
/// valid for original marching cubes vertices only (all faces around a vertex introduced later are in one voxel);
/// returns degenerate segment if the vertex is not surrounded by four voxels
LineSegm3f findLatticeEdgeEnds( const MeshTopology& t, VertId v, const Vector<VoxelId, FaceId>& face2voxel,
    const VolumeIndexer& vi, const AffineXf3f& gridToMeshXf )
{
    // every marching cubes vertex is located on a lattice edge, which is shared by exactly four voxels
    VoxelId v0, v1, v2, v3;
    for ( auto e : orgRing( t, v ) )
    {
        auto f = t.left( e );
        assert( f );
        auto vc = face2voxel[f];
        if ( vc == v0 || vc == v1 || vc == v2 )
            continue;
        if ( !v0 ) v0 = vc;
        else if ( !v1 ) v1 = vc;
        else if ( !v2 ) v2 = vc;
        else if ( !v3 )
        {
            v3 = vc;
            break;
        }
    }
    assert( v0 && v1 && v2 && v3 );
    if ( !v3 )
        return {};

    // the origins of the four voxels are the corners of a unit square orthogonal to the lattice edge
    Box3i box;
    for ( VoxelId vc : { v0, v1, v2, v3 } )
        box.include( vi.toPos( vc ) );

    // the lattice edge is directed along the only axis, where all four voxels have equal coordinate
    const auto sz = box.size();
    int dir = -1, numEqualAxes = 0;
    for ( int i = 0; i < 3; ++i )
    {
        if ( sz[i] != 0 )
            continue;
        dir = i;
        ++numEqualAxes;
    }
    assert( numEqualAxes == 1 ); // otherwise the voxels do not share one lattice edge
    if ( numEqualAxes != 1 )
        return {};

    // and it starts in the node with maximal coordinates among the voxels' origins
    auto n1 = box.max;
    ++n1[dir];
    return { gridToMeshXf( Vector3f( box.max ) ), gridToMeshXf( Vector3f( n1 ) ) };
}
}

void sharpenMarchingCubesMesh( const MeshPart& ref, Mesh& vox, Vector<VoxelId, FaceId>& face2voxel,
    const SharpenMarchingCubesMeshSettings& settings )
{
    MR_TIMER;
    assert( settings.minNewVertDev < settings.maxNewRank2VertDev );
    assert( settings.minNewVertDev < settings.maxNewRank3VertDev );
    VertNormals normals( vox.topology.vertSize() );
    VolumeIndexer vi( settings.dims );
    // find normals and correct points
    ParallelFor( normals, [&] ( VertId v )
    {
        if ( !vox.topology.hasVert( v ) )
            return;

        MeshTriPoint rp;
        Vector3f refPt;
        // the vertex is located on the lattice edge, and its position there was found by linear interpolation
        // of the values in the edge's ends; if the reference mesh crosses that edge, then the crossing
        // is the exact point, which the interpolation approximates;
        // this is valid for zero offset only, where the volume stores the distances to the reference mesh itself
        if ( settings.offset == 0 )
        {
            const auto latticeSegm = findLatticeEdgeEnds( vox.topology, v, face2voxel, vi, settings.gridToMeshXf );
            if ( latticeSegm.lengthSq() > 0 ) // otherwise no lattice edge was found for the vertex
            {
                const auto p = vox.points[v];
                const auto d = latticeSegm.dir();
                // the ray starts in the vertex and spans the edge in both directions,
                // so the crossing closest to the vertex is found
                const auto rayStart = dot( latticeSegm.a - p, d ) / d.lengthSq();
                if ( const auto isect = rayMeshIntersect( ref, Line3f{ p, d }, rayStart, rayStart + 1 ) )
                {
                    rp = isect.mtp;
                    refPt = ref.mesh.triPoint( rp );
                }
            }
        }
        if ( !rp.valid() )
        {
            const auto proj = findProjection( vox.points[v], ref );
            rp = proj.mtp;
            refPt = proj.proj.point;
        }

        Vector3f n = ( vox.points[v] - refPt ).normalized();
        Vector3f np = ref.mesh.pseudonormal( rp, ref.region );
        if ( settings.offset == 0 || n.lengthSq() <= 0 )
            n = np;
        else if ( dot( n, np ) < 0 )
            n = -n;

        if ( settings.maxOldVertPosCorrection > 0 )
        {
            const auto newPos = refPt + settings.offset * n;
            // at zero offset the reference point is the exact crossing of the vertex's lattice edge
            // with the reference mesh, so a correction of any length is trustworthy there;
            // for other offsets a large correction can be wrong and the limit is respected
            if ( settings.offset == 0 ||
                 ( newPos - vox.points[v] ).lengthSq() <= sqr( settings.maxOldVertPosCorrection ) )
                vox.points[v] = newPos;
            else
                n = Vector3f{}; //undefined
        }

        normals[v] = n;
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
        Vector3f sumDirArea; // area-weighted normal sum of the voxel's faces
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
                    const auto da = vox.dirDblArea( l ); // length = 2x face area
                    const auto a = da.length();
                    sumArea += a;
                    sumAC += a * vox.triCenter( l );
                    sumDirArea += da;
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

        if ( settings.voxelClamp )
        {
            // every marching cubes triangle is located within its own voxel, and the new vertex must not
            // break that: shorten the displacement to sharpPt, keeping its direction, until the point
            // is within the voxel's box, so the geometry of a voxel can never reach another one
            const auto pos = vi.toPos( voxel );
            Box3f voxBox;
            voxBox.include( settings.gridToMeshXf( Vector3f( pos ) + Vector3f::diagonal( 1e-3f ) ) );
            voxBox.include( settings.gridToMeshXf( Vector3f( pos ) + Vector3f::diagonal( 1.0f - 1e-3f ) ) );
            const auto shift = sharpPt - avgPt;
            float part = 1;
            for ( int i = 0; i < 3; ++i )
            {
                if ( shift[i] > 0 )
                    part = std::min( part, ( voxBox.max[i] - avgPt[i] ) / shift[i] );
                else if ( shift[i] < 0 )
                    part = std::min( part, ( voxBox.min[i] - avgPt[i] ) / shift[i] );
            }
            sharpPt = avgPt + std::max( 0.0f, part ) * shift;
        }

        const auto distSq = ( avgPt - sharpPt ).lengthSq();
        if ( distSq < sqr( settings.minNewVertDev ) )
            continue; //too little deviation of new point to introduce a vertex in mesh
        if ( rank == 2 && distSq > sqr( settings.maxNewRank2VertDev ) )
            continue; //new point is too from existing mesh triangles
        if ( rank == 3 && distSq > sqr( settings.maxNewRank3VertDev ) )
            continue; //new point is too from existing mesh triangles

        if ( !settings.voxelClamp )
        {
            // forbid in-plane shifts: the displacement to sharpPt must rise off the voxel's mean
            // surface plane, not slide along it; an in-plane sharpPt is a phantom feature and a fold source
            constexpr float minElevSin = 0.1f; // ~6 deg minimal angle between the displacement and the plane
            const Vector3f d = sharpPt - avgPt;
            if ( sqr( dot( d, sumDirArea ) ) < sqr( minElevSin ) * d.lengthSq() * sumDirArea.lengthSq() )
                continue; //sharpPt shifts (nearly) within the surface plane
        }

        auto v = vox.splitFace( f, sharpPt );
        assert( v == dirs.size() + firstNewVert );
        dirs.push_back( dir );
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
            if ( !isUnfoldQuadrangleConvex( vox.points[v], vox.points[b], vox.points[c], vox.points[d] ) )
                return false;
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
