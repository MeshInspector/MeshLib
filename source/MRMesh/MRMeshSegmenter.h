#pragma once

#include "MRSegmentMesh.h"
#include "MRMeshTopology.h"
#include "MRGraph.h"
#include "MRHeap.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRProgressCallback.h"
#include "MRTimer.h"
#include <algorithm>
#include <cfloat>
#include <optional>

namespace MR
{

/// the value to return from Policy::mergePenalty of MeshSegmenter to forbid merging of two segments
inline constexpr double prohibitMergePenalty = DBL_MAX;

/// starting from individual faces, merges them in progressively larger segments in the order of increasing merge penalty,
/// until every connected component contains only 1 segment or all remaining merges are prohibited
/// \tparam Policy must have (see SimpleSegmentPolicy as an example):
///  - type `SegmentData`, data of one segment; `a += b` must make from the data of two segments the data of their union;
///  - type `SegmentBdData`, data of the common boundary of two segments; `a += b` must make from the data of two boundary parts the data of their union;
///  - `SegmentData segmentFromTri( FaceId ) const`, returns the data of a single-triangle segment;
///  - `SegmentBdData bdFromEdge( UndirectedEdgeId ) const`, returns the data of an edge having triangles from both sides;
///  - `double mergePenalty( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd ) const` (or static), returns
///    the penalty of merging the segments (a) and (b) with the common boundary (bd), or prohibitMergePenalty to forbid the merge
template<typename Policy>
class MeshSegmenter
{
public:
    using SegmentData = typename Policy::SegmentData;
    using SegmentBdData = typename Policy::SegmentBdData;

    /// \details policy.segmentFromTri and policy.bdFromEdge are called in parallel
    static Expected<GroupOrder> run( const MeshTopology& topology, const Policy& policy, const ProgressCallback& progress = {} );

private:
    MeshSegmenter( const MeshTopology& topology, const Policy& policy ) : topology_( topology ), policy_( policy ) {}
    Expected<GroupOrder> run_( const ProgressCallback& progress );

    double penalty_( Graph::EdgeId ge ) const;

    void constructGraph_();
    void constructHeap_();
    std::optional<Graph::EndVertices> mergeNext_();

private:
    const MeshTopology& topology_;
    const Policy& policy_;

    Graph graph_;
    Vector<SegmentData, Graph::VertId> graphVertData_;
    Vector<SegmentBdData, Graph::EdgeId> graphEdgeData_;

    using Heap = MR::Heap<double, Graph::EdgeId, std::greater<double>>;
    Heap heap_;
};

template<typename Policy>
Expected<GroupOrder> MeshSegmenter<Policy>::run( const MeshTopology& topology, const Policy& policy, const ProgressCallback& progress )
{
    MR_TIMER;

    MeshSegmenter s( topology, policy );

    s.constructGraph_();
    if ( !reportProgress( progress, 0.1f ) )
        return unexpectedOperationCanceled();

    s.constructHeap_();
    if ( !reportProgress( progress, 0.2f ) )
        return unexpectedOperationCanceled();

    return s.run_( subprogress( progress, 0.2f, 1.0f ) );
}

template<typename Policy>
inline double MeshSegmenter<Policy>::penalty_( Graph::EdgeId ge ) const
{
    const auto & vv = graph_.ends( ge );
    return policy_.mergePenalty( graphVertData_[vv.v0], graphVertData_[vv.v1], graphEdgeData_[ge] );
}

template<typename Policy>
void MeshSegmenter<Policy>::constructGraph_()
{
    MR_TIMER;

    // initially:
    //   Graph::VertId = Mesh::FaceId
    //   Graph::EdgeId = Mesh::UndirectedEdgeId

    graphEdgeData_.resize( topology_.undirectedEdgeSize() );
    Graph::EndsPerEdge ends( topology_.undirectedEdgeSize() );
    Graph::EdgeBitSet validGraphEdges;
    static_cast<BitSet&>( validGraphEdges ) = topology_.findNotLoneUndirectedEdges();
    BitSetParallelFor( validGraphEdges, [&]( Graph::EdgeId ge )
    {
        UndirectedEdgeId ue( (int)ge );

        Graph::EndVertices vv;
        vv.v0 = Graph::VertId( (int)topology_.left( ue ) );
        vv.v1 = Graph::VertId( (int)topology_.right( ue ) );
        if ( !vv.v0 || !vv.v1 )
        {
            validGraphEdges.reset( ge );
            return;
        }

        assert( vv.v0 != vv.v1 );
        if ( vv.v1 < vv.v0 )
            std::swap( vv.v0, vv.v1 );
        ends[ge] = vv;
        graphEdgeData_[ge] = policy_.bdFromEdge( ue );
    } );

    graphVertData_.resize( topology_.faceSize() );
    Graph::NeighboursPerVertex neis( topology_.faceSize() );
    Graph::VertBitSet validGraphVerts;
    static_cast<BitSet&>( validGraphVerts ) = topology_.getValidFaces();
    BitSetParallelFor( validGraphVerts, [&]( Graph::VertId gv )
    {
        FaceId f( (int)gv );
        Graph::Neighbours n;
        n.reserve( topology_.getFaceDegree( f ) );
        for ( auto e : leftRing( topology_, f ) )
        {
            Graph::EdgeId ge( int( e.undirected() ) );
            assert( topology_.left( e ) == f );
            if ( topology_.right( e ) )
                n.push_back( ge );
        }
        graphVertData_[gv] = policy_.segmentFromTri( f );
        std::sort( n.begin(), n.end() );
        neis[gv] = std::move( n );
    } );
    graph_.construct( std::move( neis ), std::move( validGraphVerts ), std::move( ends ), std::move( validGraphEdges ) );
}

template<typename Policy>
void MeshSegmenter<Policy>::constructHeap_()
{
    MR_TIMER;

    std::vector<typename Heap::Element> elements( topology_.undirectedEdgeSize(), { .val = prohibitMergePenalty } );
    ParallelFor( graphEdgeData_, [&]( Graph::EdgeId ge )
    {
        elements[ge].id = ge;
        if ( !graph_.valid( ge ) )
            return;
        elements[ge].val = penalty_( ge );
    } );
    heap_ = Heap( std::move( elements ) );
}

template<typename Policy>
std::optional<Graph::EndVertices> MeshSegmenter<Policy>::mergeNext_()
{
    const auto [ge, penalty] = heap_.top();
    if ( penalty == prohibitMergePenalty )
        return std::nullopt;
    assert( graph_.valid( ge ) );
    assert( penalty == penalty_( ge ) );
    heap_.setSmallerValue( ge, prohibitMergePenalty );

    auto ends = graph_.ends( ge );
    graphVertData_[ends.v0] += graphVertData_[ends.v1];
    graph_.merge( ends.v0, ends.v1, [&]( Graph::EdgeId remnant, Graph::EdgeId dead )
        {
            graphEdgeData_[remnant] += graphEdgeData_[dead];
            heap_.setSmallerValue( dead, prohibitMergePenalty );
        } );

    for ( auto ne : graph_.neighbours( ends.v0 ) )
        heap_.setValue( ne, penalty_( ne ) );

    return ends;
}

template<typename Policy>
Expected<GroupOrder> MeshSegmenter<Policy>::run_( const ProgressCallback& progress )
{
    MR_TIMER;
    GroupOrder res;
    const auto numIters = graph_.validVerts().count(); //actually -1, but not important
    size_t iter = 0;
    while ( auto vv = mergeNext_() )
    {
        res.push_back( { FaceId( int( vv->v0 ) ), FaceId( int( vv->v1 ) ) } );
        if ( !reportProgress( progress, [&]{ return float( iter ) / numIters; }, ++iter, 1024 ) )
            return unexpectedOperationCanceled();
    }
    return res;
}

} //namespace MR
