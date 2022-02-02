#include "MRTunnelDetector.h"
#include "MRMesh.h"
#include "MREdgePaths.h"
#include "MRRegionBoundary.h"
#include "MRUnionFind.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

struct EdgeCurvature
{
    UndirectedEdgeId edge;
    float curvature = 0;

    std::pair<float, int> asSortablePair() const { return { -curvature, (int)edge }; }
    bool operator < ( const EdgeCurvature & b ) const { return asSortablePair() < b.asSortablePair(); }
};

std::vector<EdgeLoop> detectBasisTunnels( const MeshPart & mp )
{
    MR_TIMER
    
    // collect all mesh inner edges
    std::vector<EdgeCurvature> innerEdges;
    for ( EdgeId e{ 0 }; e < mp.mesh.topology.edgeSize(); e += 2 )
    {
        if ( mp.mesh.topology.isLoneEdge( e ) || !mp.mesh.topology.isInnerEdge( e, mp.region ) )
            continue;
        innerEdges.push_back( { e.undirected(), 0.0f } );
    }

    // compute curvature for every collected edge
    tbb::parallel_for(tbb::blocked_range<size_t>( 0, innerEdges.size() ), [&](const tbb::blocked_range<size_t> & range)
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            innerEdges[i].curvature = std::abs( mp.mesh.dihedralAngleSin( innerEdges[i].edge ) );
        }
    });

    // sort edges from most curved to least curved
    std::sort( innerEdges.begin(), innerEdges.end() );

    // construct maximal tree from the primary mesh edges
    UndirectedEdgeBitSet primaryTree( mp.mesh.topology.undirectedEdgeSize() );
    UnionFind<VertId> treeConnectedVertices( mp.mesh.topology.lastValidVert() + 1 );
    // here all edges that do not belong to the tree will be added (in the order from most curved to least curved)
    std::vector<EdgeCurvature> notTreeEdges;
    for ( const auto & ec : innerEdges )
    {
        const auto o = mp.mesh.topology.org( ec.edge );
        const auto d = mp.mesh.topology.dest( ec.edge );
        assert( o != d );
        if ( treeConnectedVertices.find( o ) == treeConnectedVertices.find( d ) )
        {
            // o and d are already connected by the tree, so adding this edge will introduce a loop
            notTreeEdges.push_back( ec );
            continue;
        }
        // add edge to the tree, and unite its end vertices
        treeConnectedVertices.unite( o, d );
        primaryTree.set( ec.edge );
    }

    std::vector<EdgeLoop> res;
    // construct maximal co-tree from the dual mesh edges
    UnionFind<FaceId> cotreeConnectedFace( mp.mesh.topology.lastValidFace() + 1 );

    // consider faces around each hole pre-united
    std::vector<EdgePath> bounds = findRegionBoundary( mp.mesh.topology, mp.region );
    for ( const auto & loop : bounds )
    {
        if ( loop.empty() )
            continue;
        FaceId first;
        for ( auto e : loop )
        {
            auto l = mp.mesh.topology.left( e );
            if ( !l )
                continue;
            if ( first )
                cotreeConnectedFace.unite( first, l );
            else
                first = l;
        }
    }

    // go in the order from least curved to most curved
    for ( int i = (int)notTreeEdges.size() - 1; i >= 0; --i )
    {
        const auto & ec = notTreeEdges[i];
        const auto l = mp.mesh.topology.left( ec.edge );
        const auto r = mp.mesh.topology.right( ec.edge );
        assert( l && r && l != r );
        if ( cotreeConnectedFace.find( l ) == cotreeConnectedFace.find( r ) )
        {
            // l and r are already connected by the co-tree, so adding this edge will introduce a loop
            const auto o = mp.mesh.topology.org( ec.edge );
            const auto d = mp.mesh.topology.dest( ec.edge );
            assert( o != d );
            assert( !primaryTree.test( ec.edge ) );
            assert( treeConnectedVertices.find( o ) == treeConnectedVertices.find( d ) );

            const float numEdges = float( mp.mesh.topology.undirectedEdgeSize() ); // a value large than any loop length in edges
            auto treeMetric = [numEdges, &primaryTree]( EdgeId e )
            {
                return primaryTree.test( e.undirected() ) ? 1.0f : numEdges;
            };
            auto tunnel = buildSmallestMetricPath( mp.mesh.topology, treeMetric, d, o );
            tunnel.push_back( ec.edge );
            assert( isEdgeLoop( mp.mesh.topology, tunnel ) );
            res.push_back( std::move( tunnel ) );
            continue;
        }
        // add edge to the co-tree
        cotreeConnectedFace.unite( l, r );
    }

    return res;
}


FaceBitSet detectTunnelFaces( const MeshPart & mp, float maxTunnelLength )
{
    MR_TIMER;
    FaceBitSet activeRegion = mp.mesh.topology.getFaceIds( mp.region );
    MeshPart activeMeshPart{ mp.mesh, &activeRegion };
    FaceBitSet tunnelFaces;
    VertBitSet tunnelVerts( mp.mesh.topology.lastValidVert() + 1 );

    for ( ;; )
    {
        auto basisTunnels = detectBasisTunnels( activeMeshPart );
        const auto numBasisTunnels = basisTunnels.size();

        sortPathsByLength( basisTunnels, mp.mesh );
        for ( int i = 0; i < basisTunnels.size(); ++i )
        {
            if ( calcPathLength( basisTunnels[i], mp.mesh ) > maxTunnelLength )
            {
                basisTunnels.erase( basisTunnels.begin() + i, basisTunnels.end() );
                break;
            }
        }
        if ( basisTunnels.empty() )
            break;

        tunnelVerts.set( 0_v, tunnelVerts.size(), false );
        int numSelectedTunnels = 0;
        for ( const auto & t : basisTunnels )
        {
            bool touchAlreadySelectedTunnel = false;
            for ( EdgeId e : t )
            {
                if ( tunnelVerts.test( mp.mesh.topology.org( e ) ) )
                {
                    touchAlreadySelectedTunnel = true;
                    break;
                }
            }
            if ( touchAlreadySelectedTunnel )
                continue;
            ++numSelectedTunnels;

            for ( EdgeId e : t )
            {
                tunnelVerts.set( mp.mesh.topology.org( e ) );
            }
            addLeftBand( mp.mesh.topology, t, tunnelFaces );
        }
        activeRegion -= tunnelFaces;
        assert( numSelectedTunnels > 0 );
        if ( numSelectedTunnels >= numBasisTunnels )
            break; // maximal not-intersection set of tunnels has been used
    }
    return tunnelFaces;
}

} //namespace MR
