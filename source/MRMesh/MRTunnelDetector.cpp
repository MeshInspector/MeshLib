#include "MRTunnelDetector.h"
#include "MRMesh.h"
#include "MREdgePaths.h"
#include "MRRegionBoundary.h"
#include "MRUnionFind.h"
#include "MRTimer.h"
#include "MRExpected.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

struct EdgeCurvature
{
    UndirectedEdgeId edge;
    float metric = 0;

    std::pair<float, int> asSortablePair() const { return { metric, (int)edge }; }
    bool operator < ( const EdgeCurvature & b ) const { return asSortablePair() < b.asSortablePair(); }
};

class BasisTunnelsDetector
{
public:
    BasisTunnelsDetector( const MeshPart & mp, EdgeMetric metric );
    VoidOrErrStr prepare( ProgressCallback cb );
    tl::expected<std::vector<EdgeLoop>, std::string> detect( ProgressCallback cb );

private:
    std::vector<EdgeCurvature> innerEdges_; // sorted by metric
    const MeshPart & mp_;
    EdgeMetric metric_;
};

BasisTunnelsDetector::BasisTunnelsDetector( const MeshPart & mp, EdgeMetric metric )
    : mp_( mp )
    , metric_( std::move( metric ) )
{
    assert( metric_ );
}

VoidOrErrStr BasisTunnelsDetector::prepare( ProgressCallback cb )
{
    MR_TIMER

    // count inner edges
    size_t numInnerEdges = 0;
    for ( EdgeId e{ 0 }; e < mp_.mesh.topology.edgeSize(); e += 2 )
    {
        if ( mp_.mesh.topology.isLoneEdge( e ) || !mp_.mesh.topology.isInnerEdge( e, mp_.region ) )
            continue;
        ++numInnerEdges;
    }

    if ( !reportProgress( cb, 0.25f ) )
        return tl::make_unexpected( "Operation was canceled" );

    // collect all mesh inner edges
    innerEdges_.clear();
    innerEdges_.reserve( numInnerEdges );
    for ( EdgeId e{ 0 }; e < mp_.mesh.topology.edgeSize(); e += 2 )
    {
        if ( mp_.mesh.topology.isLoneEdge( e ) || !mp_.mesh.topology.isInnerEdge( e, mp_.region ) )
            continue;
        innerEdges_.push_back( { e.undirected(), 0.0f } );
    }
    assert( innerEdges_.size() == numInnerEdges );

    if ( !reportProgress( cb, 0.5f ) )
        return tl::make_unexpected( "Operation was canceled" );

    // compute curvature for every collected edge
    tbb::parallel_for(tbb::blocked_range<size_t>( 0, innerEdges_.size() ), [&](const tbb::blocked_range<size_t> & range)
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            innerEdges_[i].metric = metric_( innerEdges_[i].edge );
        }
    });

    if ( !reportProgress( cb, 0.75f ) )
        return tl::make_unexpected( "Operation was canceled" );

    // sort edges from most curved to least curved
    tbb::parallel_sort( innerEdges_.begin(), innerEdges_.end() );
    return {};
}

tl::expected<std::vector<EdgeLoop>, std::string> BasisTunnelsDetector::detect( ProgressCallback cb )
{
    MR_TIMER

    if ( !reportProgress( cb, 0.0f ) )
        return tl::make_unexpected( "Operation was canceled" );

    // construct maximal tree from the primary mesh edges
    UndirectedEdgeBitSet primaryTree( mp_.mesh.topology.undirectedEdgeSize() );
    UnionFind<VertId> treeConnectedVertices( mp_.mesh.topology.lastValidVert() + 1 );
    // here all edges that do not belong to the tree will be added (in the order from most curved to least curved)
    std::vector<EdgeCurvature> notTreeEdges;
    for ( size_t i = 0; i < innerEdges_.size(); ++i)
    {
        const auto& ec = innerEdges_[i];
        const auto o = mp_.mesh.topology.org( ec.edge );
        const auto d = mp_.mesh.topology.dest( ec.edge );
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

    if ( !reportProgress( cb, 0.25f ) )
        return tl::make_unexpected( "Operation was canceled" );

    // construct maximal co-tree from the dual mesh edges
    UnionFind<FaceId> cotreeConnectedFace( mp_.mesh.topology.lastValidFace() + 1 );

    // consider faces around each hole pre-united
    std::vector<EdgePath> bounds = findLeftBoundary( mp_.mesh.topology, mp_.region );

    for ( size_t i = 0; i < bounds.size(); ++i )
    {
        const auto& loop = bounds[i];
        if ( loop.empty() )
            continue;
        FaceId first;
        for ( auto e : loop )
        {
            auto l = mp_.mesh.topology.left( e );
            if ( !l )
                continue;
            if ( first )
                cotreeConnectedFace.unite( first, l );
            else
                first = l;
        }
    }

    if ( !reportProgress( cb, 0.5f ) )
        return tl::make_unexpected( "Operation was canceled" );

    std::vector<EdgeId> joinEdges;
    for ( int i = (int)notTreeEdges.size() - 1; i >= 0; --i )
    {
        const auto & ec = notTreeEdges[i];
        const auto l = mp_.mesh.topology.left( ec.edge );
        const auto r = mp_.mesh.topology.right( ec.edge );
        assert( l && r && l != r );
        if ( cotreeConnectedFace.find( l ) == cotreeConnectedFace.find( r ) )
        {
            // l and r are already connected by the co-tree, so adding this edge will introduce a loop
            joinEdges.push_back( ec.edge );
            continue;
        }
        // add edge to the co-tree
        cotreeConnectedFace.unite( l, r );
    }

    if ( !reportProgress( cb, 0.75f ) )
        return tl::make_unexpected( "Operation was canceled" );

    std::vector<EdgeLoop> res( joinEdges.size() );
    const float numEdges = float( mp_.mesh.topology.undirectedEdgeSize() ); // a value larger than any loop length in edges
    auto treeMetric = [numEdges, &primaryTree]( EdgeId e )
    {
        return primaryTree.test( e.undirected() ) ? 1.0f : numEdges;
    };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, res.size() ), [&]( const tbb::blocked_range<size_t> & range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto edge = joinEdges[i];
            const auto o = mp_.mesh.topology.org( edge );
            const auto d = mp_.mesh.topology.dest( edge );
            assert( o != d );
            assert( !primaryTree.test( edge ) );
            assert( treeConnectedVertices.find( o ) == treeConnectedVertices.find( d ) );

            auto tunnel = buildSmallestMetricPath( mp_.mesh.topology, treeMetric, d, o );
            tunnel.push_back( edge );
            assert( isEdgeLoop( mp_.mesh.topology, tunnel ) );
            res[i] = std::move( tunnel );
        }
    });

    return res;
}

} //anonymous namespace

tl::expected<std::vector<EdgeLoop>, std::string> detectBasisTunnels( const MeshPart & mp, EdgeMetric metric, ProgressCallback cb )
{
    MR_TIMER
    if ( !metric )
        metric = discreteMinusAbsMeanCurvatureMetric( mp.mesh );

    BasisTunnelsDetector d( mp, metric );
    if ( auto v = d.prepare( subprogress( cb, 0.0f, 0.25f ) ); !v.has_value() )
        return tl::make_unexpected( std::move( v.error() ) );
    return d.detect( subprogress( cb, 0.25f, 1.0f ) );
}

tl::expected<FaceBitSet, std::string> detectTunnelFaces( const MeshPart & mp, float maxTunnelLength, const EdgeMetric & metric, ProgressCallback progressCallback )
{
    MR_TIMER;
    FaceBitSet activeRegion = mp.mesh.topology.getFaceIds( mp.region );
    MeshPart activeMeshPart{ mp.mesh, &activeRegion };
    FaceBitSet tunnelFaces;
    VertBitSet tunnelVerts( mp.mesh.topology.lastValidVert() + 1 );

    float initialProgress = 0.0f;
    float targetProgress = 0.49f;

    for ( ;; )
    {
        auto basisTunnels = detectBasisTunnels( activeMeshPart, metric, MR::subprogress(progressCallback, initialProgress, targetProgress) );
        if ( !basisTunnels.has_value() )
            return tl::make_unexpected( basisTunnels.error() );

        const auto numBasisTunnels = basisTunnels->size();

        sortPathsByLength( *basisTunnels, mp.mesh );
        for ( int i = 0; i < basisTunnels->size(); ++i )
        {
            if ( calcPathLength( (*basisTunnels)[i], mp.mesh ) > maxTunnelLength )
            {
                basisTunnels->erase( basisTunnels->begin() + i, basisTunnels->end() );
                break;
            }
        }
        if ( basisTunnels->empty() )
            break;

        tunnelVerts.set( 0_v, tunnelVerts.size(), false );
        int numSelectedTunnels = 0;
        for ( const auto & t : *basisTunnels )
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
        if ( progressCallback && !progressCallback( targetProgress + 0.01f ) )
            return tl::make_unexpected( "Operation was canceled" );

        initialProgress = targetProgress + 0.01f;
        targetProgress = ( ( initialProgress  + 1.0f ) * 0.5f ) - 0.01f;

        if ( numSelectedTunnels >= numBasisTunnels )
            break; // maximal not-intersection set of tunnels has been used
    }

    return tunnelFaces;
}

} //namespace MR
