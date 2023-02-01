#include "MRSharpenMarchingCubesMesh.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRTriMath.h"

namespace MR
{

void sharpenMarchingCubesMesh( const Mesh & ref, Mesh & vox, Vector<VoxelId, FaceId> & face2voxel,
    const SharpenMarchingCubesMeshSettings & settings )
{
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
            normals[v] = n;

            if ( settings.correctOldVertPos )
                vox.points[v] = proj.proj.point + settings.offset * n;
        }
    } );

    auto facesToProcess = vox.topology.getValidFaces();
    VertId firstNewVert( vox.topology.vertSize() );
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
        auto sharpPt = pacc.findBestCrossPoint( avgPt );
        if ( ( avgPt - sharpPt ).lengthSq() <= sqr( settings.newVertDev ) )
            continue; //too little deviation of new point to introduce a vertex in mesh

        auto v = vox.splitFace( f );
        assert( v >= firstNewVert );
        vox.points.autoResizeSet( v, sharpPt );
        for ( auto ei : orgRing( vox.topology, v ) )
            face2voxel.autoResizeSet( vox.topology.left( ei ), voxel );
        // connect new vertex with every vertex from the voxel
        vox.topology.flipEdgesAround( v, [&]( EdgeId e )
        {
            auto r = vox.topology.right( e );
            assert( r );
            if ( face2voxel[r] == voxel )
            {
                auto b = vox.topology.dest( vox.topology.prev( e ) );
                for ( auto ei : orgRing( vox.topology, vox.topology.next( e ).sym() ) )
                {
                    assert( vox.topology.org( ei ) == v );
                    if ( vox.topology.dest( ei ) == b )
                        return false;
                }
                return true;
            }
            return false;
        } );
    }

    // find edges between voxels with new vertices
    UndirectedEdgeBitSet sharpEdges( vox.topology.undirectedEdgeSize() );
    for ( auto v = firstNewVert; v < vox.topology.vertSize(); ++v )
    {
        for ( auto ei : orgRing( vox.topology, v ) )
        {
            if ( !vox.topology.left( ei ) )
                continue;
            EdgeId e = vox.topology.prev( ei.sym() );
            if ( !vox.topology.right( e ) )
                continue;
            auto b = vox.topology.dest( vox.topology.prev( e ) );
            if ( b > v )
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
                    sharpEdges.set( e.undirected() );
            }
        }
    }

    // flip edges between voxels with new vertices to form sharp ridges
    for ( EdgeId e : sharpEdges )
    {
        bool good = true;
        auto b = vox.topology.dest( vox.topology.prev( e ) );
        for ( auto ei : orgRing( vox.topology, vox.topology.next( e ).sym() ) )
        {
            if ( vox.topology.dest( ei ) == b )
            {
                good = false;
                break;
            }
        }
        if ( good )
            vox.topology.flipEdge( e );
    }

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );
}

} //namespace MR
