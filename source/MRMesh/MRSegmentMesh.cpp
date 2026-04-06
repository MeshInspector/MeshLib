#include "MRSegmentMesh.h"
#include "MRMesh.h"
#include "MRGraph.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRHeap.h"
#include "MRTimer.h"
#include <cfloat>
#include <optional>

namespace MR
{

namespace
{

static constexpr double NoEdgeMetric = DBL_MAX;

class MeshSegmenter
{
public:
    MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric );
    GroupOrder run();

private:
    double vertMetricAfterMerge_( Graph::EdgeId ge ) const;

    void constructGraph_();
    void constructHeap_();
    std::optional<Graph::EndVertices> mergeNext_();

private:
    const MeshTopology& topology_;
    const EdgeMetric& metric_;

    Graph graph_;
    Vector<double, Graph::VertId> graphVertMetrics_;
    Vector<double, Graph::EdgeId> graphEdgeMetrics_;

    using Heap = MR::Heap<double, GraphEdgeId, std::greater<double>>;
    Heap heap_;
};

MeshSegmenter::MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric )
    : topology_( topology ), metric_( metric )
{
    constructGraph_();
    constructHeap_();
}

inline double MeshSegmenter::vertMetricAfterMerge_( Graph::EdgeId ge ) const
{
    const auto & vv = graph_.ends( ge );
    return graphVertMetrics_[vv.v0] + graphVertMetrics_[vv.v1] - 2 * graphEdgeMetrics_[ge];
}

void MeshSegmenter::constructGraph_()
{
    MR_TIMER;

    // initially:
    //   Graph::VertId = Mesh::FaceId
    //   Graph::EdgeId = Mesh::UndirectedEdgeId

    graphEdgeMetrics_.resize( topology_.undirectedEdgeSize() );
    Graph::EndsPerEdge ends( topology_.undirectedEdgeSize() );
    ParallelFor( graphEdgeMetrics_, [&]( Graph::EdgeId ge )
    {
        UndirectedEdgeId ue( (int)ge );
        if ( topology_.isLoneEdge( ue ) )
            return;
        graphEdgeMetrics_[ge] = metric_( ue );

        Graph::EndVertices vv;
        vv.v0 = Graph::VertId( (int)topology_.left( ue ) );
        vv.v1 = Graph::VertId( (int)topology_.right( ue ) );
        if ( vv.v0 && vv.v1 )
        {
            assert( vv.v0 != vv.v1 );
            if ( vv.v1 < vv.v0 )
                std::swap( vv.v0, vv.v1 );
            ends[ge] = vv;
        }
    } );

    graphVertMetrics_.resize( topology_.faceSize() );
    Graph::NeighboursPerVertex neis( topology_.faceSize() );
    ParallelFor( graphVertMetrics_, [&]( Graph::VertId gv )
    {
        FaceId f( (int)gv );
        if ( !topology_.hasFace( f ) )
            return;

        Graph::Neighbours n;
        n.reserve( topology_.getFaceDegree( f ) );
        double m = 0;
        for ( auto e : leftRing( topology_, f ) )
        {
            Graph::EdgeId ge( int( e.undirected() ) );
            assert( topology_.left( e ) = f );
            if ( topology_.right( e ) )
                n.push_back( ge );
            m += graphEdgeMetrics_[ge];
        }
        graphVertMetrics_[gv] = m;
        neis[gv] = std::move( n );
    } );
    graph_.construct( std::move( neis ), std::move( ends ) );
}

void MeshSegmenter::constructHeap_()
{
    MR_TIMER;

    std::vector<Heap::Element> elements( topology_.undirectedEdgeSize(), { .val = NoEdgeMetric } );
    ParallelFor( graphEdgeMetrics_, [&]( Graph::EdgeId ge )
    {
        elements[ge].id = ge;
        if ( !graph_.valid( ge ) )
            return;
        elements[ge].val = vertMetricAfterMerge_( ge );
    } );
    heap_ = Heap( std::move( elements ) );
}

std::optional<Graph::EndVertices> MeshSegmenter::mergeNext_()
{
    const auto [ge, metric] = heap_.top();
    if ( metric == NoEdgeMetric )
        return std::nullopt;
    assert( graph_.valid( ge ) );
    assert( metric == vertMetricAfterMerge_( ge ) );
    heap_.setSmallerValue( ge, NoEdgeMetric );

    auto ends = graph_.ends( ge );
    graphVertMetrics_[ends.v0] = metric;
    graph_.merge( ends.v0, ends.v1, [&]( Graph::EdgeId remnant, Graph::EdgeId dead )
        {
            graphEdgeMetrics_[remnant] += graphEdgeMetrics_[dead];
            heap_.setValue( remnant, vertMetricAfterMerge_( remnant ) );
            heap_.setSmallerValue( dead, NoEdgeMetric );
        } );

    return ends;
}

GroupOrder MeshSegmenter::run()
{
    MR_TIMER;
    GroupOrder res;
    while ( auto vv = mergeNext_() )
        res.push_back( { FaceId( int( vv->v0 ) ), FaceId( int( vv->v1 ) ) } );
    return res;
}

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const MeshTopology& topology, const EdgeMetric& metric )
{
    MR_TIMER;
    if ( !metric )
        return unexpected( "no metric given" );

    MeshSegmenter s( topology, metric );
    return s.run();
}

} //namespace MR
