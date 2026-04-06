#include "MRSegmentMesh.h"
#include "MRMesh.h"
#include "MRGraph.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

class MeshSegmenter
{
public:
    MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric );

private:
    void constructGraph_();

private:
    const MeshTopology& topology_;
    const EdgeMetric& metric_;

    Graph graph_;
    Vector<double, Graph::VertId> graphVertMetrics_;
    Vector<double, Graph::EdgeId> graphEdgeMetrics_;
};

MeshSegmenter::MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric )
    : topology_( topology ), metric_( metric )
{
    constructGraph_();
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

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const MeshTopology& topology, const EdgeMetric& metric )
{
    MR_TIMER;
    if ( !metric )
        return unexpected( "no metric given" );

    MeshSegmenter s( topology, metric );

    GroupOrder res;
    return res;
}

} //namespace MR
