#include "MRTunnelDetector.h"
#include "MRMesh.h"
#include "MREdgePaths.h"
#include "MRInTreePathBuilder.h"
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
    Expected<void> prepare( ProgressCallback cb );
    // after prepare(...) region can only shrink, not become larger
    Expected<std::vector<EdgeLoop>> detect( ProgressCallback cb );

private:
    std::vector<EdgeCurvature> innerEdges_; // sorted by metric
    const MeshPart & mp_;
    EdgeMetric metric_;

    UndirectedEdgeBitSet primaryTree_;
    UnionFind<VertId> treeConnectedVertices_;
    UnionFind<FaceId> cotreeConnectedFace_;
};

BasisTunnelsDetector::BasisTunnelsDetector( const MeshPart & mp, EdgeMetric metric )
    : mp_( mp )
    , metric_( std::move( metric ) )
{
    assert( metric_ );
}

Expected<void> BasisTunnelsDetector::prepare( ProgressCallback cb )
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
        return unexpectedOperationCanceled();

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
        return unexpectedOperationCanceled();

    // compute curvature for every collected edge
    tbb::parallel_for(tbb::blocked_range<size_t>( 0, innerEdges_.size() ), [&](const tbb::blocked_range<size_t> & range)
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            innerEdges_[i].metric = metric_( innerEdges_[i].edge );
        }
    });

    if ( !reportProgress( cb, 0.75f ) )
        return unexpectedOperationCanceled();

    // sort edges from most curved to least curved
    tbb::parallel_sort( innerEdges_.begin(), innerEdges_.end() );
    return {};
}

Expected<std::vector<EdgeLoop>> BasisTunnelsDetector::detect( ProgressCallback cb )
{
    MR_TIMER

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    // construct maximal tree from the primary mesh edges
    primaryTree_.clear();
    primaryTree_.resize( mp_.mesh.topology.undirectedEdgeSize() );
    treeConnectedVertices_.reset( mp_.mesh.topology.lastValidVert() + 1 );
    for ( size_t i = 0; i < innerEdges_.size(); ++i)
    {
        const auto& ec = innerEdges_[i];
        if ( !ec.edge )
            continue;
        if ( !mp_.mesh.topology.isInnerEdge( ec.edge, mp_.region ) )
        {
            // region can only shrink, so some more edges become not-inner
            innerEdges_[i] = {}; // invalidate such edges
            continue; 
        }

        const auto o = mp_.mesh.topology.org( ec.edge );
        const auto d = mp_.mesh.topology.dest( ec.edge );
        assert( o != d );
        if ( !treeConnectedVertices_.unite( o, d ).second )
        {
            // o and d are already connected by the tree, so adding this edge will introduce a loop
            continue;
        }
        // now o and d are united; add edge to the tree
        primaryTree_.set( ec.edge );
    }

    if ( !reportProgress( cb, 0.25f ) )
        return unexpectedOperationCanceled();

    // construct maximal co-tree from the dual mesh edges
    cotreeConnectedFace_.reset( mp_.mesh.topology.lastValidFace() + 1 );

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
                cotreeConnectedFace_.unite( first, l );
            else
                first = l;
        }
    }

    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    std::vector<EdgeId> joinEdges;
    // check all edges not from primary tree, and build co-tree
    for ( int i = (int)innerEdges_.size() - 1; i >= 0; --i )
    {
        const auto & ec = innerEdges_[i];
        if ( !ec.edge )
            continue;
        if ( primaryTree_.test( ec.edge ) )
            continue;
        const auto l = mp_.mesh.topology.left( ec.edge );
        const auto r = mp_.mesh.topology.right( ec.edge );
        assert( l && r && l != r );
        if ( !cotreeConnectedFace_.unite( l, r ).second )
        {
            // l and r are already connected by the co-tree, so adding this edge will introduce a loop
            joinEdges.push_back( ec.edge );
            continue;
        }
    }

    if ( !reportProgress( cb, 0.75f ) )
        return unexpectedOperationCanceled();

    std::vector<EdgeLoop> res( joinEdges.size() );
    InTreePathBuilder inTreePathBuilder( mp_.mesh.topology, primaryTree_ );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, res.size() ), [&]( const tbb::blocked_range<size_t> & range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto edge = joinEdges[i];
            const auto o = mp_.mesh.topology.org( edge );
            const auto d = mp_.mesh.topology.dest( edge );
            assert( o != d );
            assert( !primaryTree_.test( edge ) );
            assert( treeConnectedVertices_.find( o ) == treeConnectedVertices_.find( d ) );

            auto tunnel = inTreePathBuilder.build( d, o );
            tunnel.push_back( edge );
            assert( isEdgeLoop( mp_.mesh.topology, tunnel ) );
            res[i] = std::move( tunnel );
        }
    });

    return res;
}

} //anonymous namespace

Expected<std::vector<EdgeLoop>> detectBasisTunnels( const MeshPart & mp, EdgeMetric metric, ProgressCallback cb )
{
    MR_TIMER
    if ( !metric )
        metric = discreteMinusAbsMeanCurvatureMetric( mp.mesh );

    BasisTunnelsDetector d( mp, metric );
    if ( auto v = d.prepare( subprogress( cb, 0.0f, 0.25f ) ); !v.has_value() )
        return unexpected( std::move( v.error() ) );
    return d.detect( subprogress( cb, 0.25f, 1.0f ) );
}

Expected<FaceBitSet> detectTunnelFaces( const MeshPart & mp, const DetectTunnelSettings & settings )
{
    MR_TIMER
    auto metric = settings.metric;
    if ( !metric )
        metric = discreteMinusAbsMeanCurvatureMetric( mp.mesh );

    FaceBitSet activeRegion = mp.mesh.topology.getFaceIds( mp.region );
    MeshPart activeMeshPart{ mp.mesh, &activeRegion };
    FaceBitSet tunnelFaces;
    VertBitSet tunnelVerts( mp.mesh.topology.lastValidVert() + 1 );

    BasisTunnelsDetector d( activeMeshPart, metric );
    if ( auto v = d.prepare( subprogress( settings.progress, 0.0f, 0.33f ) ); !v.has_value() )
        return unexpected( std::move( v.error() ) );

    float initialProgress = 0.33f;
    float targetProgress = 0.66f;

    for ( int iter = 0; iter < settings.maxIters; ++iter )
    {
        auto basisTunnels = d.detect( MR::subprogress( settings.progress, initialProgress, targetProgress ) );
        if ( !basisTunnels.has_value() )
            return unexpected( basisTunnels.error() );

        const auto numBasisTunnels = basisTunnels->size();

        sortPathsByLength( *basisTunnels, mp.mesh );
        for ( int i = 0; i < basisTunnels->size(); ++i )
        {
            if ( calcPathLength( (*basisTunnels)[i], mp.mesh ) > settings.maxTunnelLength )
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
        activeRegion -= tunnelFaces; // reduce region
        assert( numSelectedTunnels > 0 );
        if ( !reportProgress( settings.progress, targetProgress + 0.01f ) )
            return unexpectedOperationCanceled();

        initialProgress = targetProgress + 0.01f;
        targetProgress = ( ( initialProgress  + 1.0f ) * 0.5f ) - 0.01f;

        if ( numSelectedTunnels >= numBasisTunnels )
            break; // maximal not-intersection set of tunnels has been used
    }

    return tunnelFaces;
}

} //namespace MR
