#include "MRSegmentMesh.h"
#include "MRMesh.h"
#include "MRGraph.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRHeap.h"
#include "MRUnionFind.h"
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
    MeshSegmenter( const Mesh& mesh, const EdgeMetric& lengthCurvMetric );
    GroupOrder run();

private:
    double vertMetricAfterMerge_( Graph::EdgeId ge ) const;

    void constructGraph_();
    void constructHeap_();
    std::optional<Graph::EndVertices> mergeNext_();

private:
    const Mesh& mesh_;
    const EdgeMetric& lengthCurvMetric_;

    Graph graph_;
    Vector<double, Graph::VertId> graphVertMetrics_;
    Vector<double, Graph::EdgeId> graphEdgeMetrics_;

    using Heap = MR::Heap<double, GraphEdgeId, std::greater<double>>;
    Heap heap_;
};

MeshSegmenter::MeshSegmenter( const Mesh& mesh, const EdgeMetric& lengthCurvMetric )
    : mesh_( mesh ), lengthCurvMetric_( lengthCurvMetric )
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

    graphEdgeMetrics_.resize( mesh_.topology.undirectedEdgeSize() );
    Graph::EndsPerEdge ends( mesh_.topology.undirectedEdgeSize() );
    ParallelFor( graphEdgeMetrics_, [&]( Graph::EdgeId ge )
    {
        UndirectedEdgeId ue( (int)ge );
        if ( mesh_.topology.isLoneEdge( ue ) )
            return;
        graphEdgeMetrics_[ge] = lengthCurvMetric_( ue );

        Graph::EndVertices vv;
        vv.v0 = Graph::VertId( (int)mesh_.topology.left( ue ) );
        vv.v1 = Graph::VertId( (int)mesh_.topology.right( ue ) );
        if ( vv.v0 && vv.v1 )
        {
            assert( vv.v0 != vv.v1 );
            if ( vv.v1 < vv.v0 )
                std::swap( vv.v0, vv.v1 );
            ends[ge] = vv;
        }
    } );

    graphVertMetrics_.resize( mesh_.topology.faceSize() );
    Graph::NeighboursPerVertex neis( mesh_.topology.faceSize() );
    ParallelFor( graphVertMetrics_, [&]( Graph::VertId gv )
    {
        FaceId f( (int)gv );
        if ( !mesh_.topology.hasFace( f ) )
            return;

        Graph::Neighbours n;
        n.reserve( mesh_.topology.getFaceDegree( f ) );
        double m = 0;
        for ( auto e : leftRing( mesh_.topology, f ) )
        {
            Graph::EdgeId ge( int( e.undirected() ) );
            assert( mesh_.topology.left( e ) = f );
            if ( mesh_.topology.right( e ) )
                n.push_back( ge );
            m += graphEdgeMetrics_[ge];
        }
        graphVertMetrics_[gv] = m;
        std::sort( n.begin(), n.end() );
        neis[gv] = std::move( n );
    } );
    graph_.construct( std::move( neis ), std::move( ends ) );
}

void MeshSegmenter::constructHeap_()
{
    MR_TIMER;

    std::vector<Heap::Element> elements( mesh_.topology.undirectedEdgeSize(), { .val = NoEdgeMetric } );
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
    const auto [ge, lengthCurvMetric] = heap_.top();
    if ( lengthCurvMetric == NoEdgeMetric )
        return std::nullopt;
    assert( graph_.valid( ge ) );
    assert( lengthCurvMetric == vertMetricAfterMerge_( ge ) );
    heap_.setSmallerValue( ge, NoEdgeMetric );

    auto ends = graph_.ends( ge );
    graphVertMetrics_[ends.v0] = lengthCurvMetric;
    graph_.merge( ends.v0, ends.v1, [&]( Graph::EdgeId remnant, Graph::EdgeId dead )
        {
            graphEdgeMetrics_[remnant] += graphEdgeMetrics_[dead];
            heap_.setSmallerValue( dead, NoEdgeMetric );
        } );

    for ( auto ne : graph_.neighbours( ends.v0 ) )
        heap_.setValue( ne, vertMetricAfterMerge_( ne ) );

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

Expected<GroupOrder> segmentMesh( const Mesh& mesh, const EdgeMetric& lengthCurvMetric )
{
    MR_TIMER;
    if ( !lengthCurvMetric )
        return unexpected( "no lengthCurvMetric given" );

    MeshSegmenter s( mesh, lengthCurvMetric );
    return s.run();
}

UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments )
{
    MR_TIMER;
    int numMerges = topology.numValidFaces() - numSegments;
    assert( numMerges >= 0 );
    assert( numMerges <= groupOrder.size() );
    numMerges = std::clamp( numMerges, 0, (int)groupOrder.size() );

    UnionFind<FaceId> unionFindStruct( topology.faceSize() );
    for ( int i = 0; i < numMerges; ++i )
    {
       [[maybe_unused]] auto p = unionFindStruct.unite( groupOrder[i].aFace, groupOrder[i].bFace );
       assert( p.second );
    }
    const auto & roots = unionFindStruct.roots();

    UndirectedEdgeBitSet res( topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        const auto l = topology.left( ue );
        if ( !l )
            return;
        const auto r = topology.right( ue );
        if ( !r )
            return;
        if ( roots[l] != roots[r] )
            res.set( ue );
    } );
    return res;
}

} //namespace MR
