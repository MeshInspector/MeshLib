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
#include <cfloat>
#include <algorithm>
#include <optional>

namespace MR
{

/// the value to return from MergePenalty of MeshSegmenter to forbid merging of two segments
inline constexpr double prohibitMergePenalty = DBL_MAX;

/// starting from individual faces, merges them in progressively larger segments in the order of increasing merge penalty,
/// until every connected component contains only 1 segment or all remaining merges are prohibited
/// \tparam SegmentData data of one segment; `a += b` must make from the data of two segments the data of their union
/// \tparam SegmentBdData data of the common boundary of two segments; `a += b` must make from the data of two boundary parts the data of their union
/// \tparam MergePenalty functor `double( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd )` returning
///         the penalty of merging the segments (a) and (b) with the common boundary (bd), or prohibitMergePenalty to forbid the merge
template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
class MeshSegmenter
{
public:
    /// \param faceData functor `SegmentData( FaceId )` returning the data of a single-face segment
    /// \param edgeData functor `SegmentBdData( UndirectedEdgeId )` returning the data of an edge having faces from both sides
    /// \details both faceData and edgeData are called in parallel
    template<typename FaceData, typename EdgeData>
    static Expected<GroupOrder> run( const MeshTopology& topology, FaceData&& faceData, EdgeData&& edgeData,
        MergePenalty mergePenalty, const ProgressCallback& progress = {} );

private:
    MeshSegmenter( const MeshTopology& topology, MergePenalty&& mergePenalty ) : topology_( topology ), mergePenalty_( std::move( mergePenalty ) ) {}
    Expected<GroupOrder> run_( const ProgressCallback& progress );

    double penalty_( Graph::EdgeId ge ) const;

    template<typename FaceData, typename EdgeData>
    void constructGraph_( FaceData&& faceData, EdgeData&& edgeData );
    void constructHeap_();
    std::optional<Graph::EndVertices> mergeNext_();

private:
    const MeshTopology& topology_;
    MergePenalty mergePenalty_;

    Graph graph_;
    Vector<SegmentData, Graph::VertId> graphVertData_;
    Vector<SegmentBdData, Graph::EdgeId> graphEdgeData_;

    using Heap = MR::Heap<double, Graph::EdgeId, std::greater<double>>;
    Heap heap_;
};

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
template<typename FaceData, typename EdgeData>
Expected<GroupOrder> MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::run( const MeshTopology& topology,
    FaceData&& faceData, EdgeData&& edgeData, MergePenalty mergePenalty, const ProgressCallback& progress )
{
    MR_TIMER;

    MeshSegmenter s( topology, std::move( mergePenalty ) );

    s.constructGraph_( faceData, edgeData );
    if ( !reportProgress( progress, 0.1f ) )
        return unexpectedOperationCanceled();

    s.constructHeap_();
    if ( !reportProgress( progress, 0.2f ) )
        return unexpectedOperationCanceled();

    return s.run_( subprogress( progress, 0.2f, 1.0f ) );
}

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
inline double MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::penalty_( Graph::EdgeId ge ) const
{
    const auto & vv = graph_.ends( ge );
    return mergePenalty_( graphVertData_[vv.v0], graphVertData_[vv.v1], graphEdgeData_[ge] );
}

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
template<typename FaceData, typename EdgeData>
void MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::constructGraph_( FaceData&& faceData, EdgeData&& edgeData )
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
        graphEdgeData_[ge] = edgeData( ue );
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
        graphVertData_[gv] = faceData( f );
        std::sort( n.begin(), n.end() );
        neis[gv] = std::move( n );
    } );
    graph_.construct( std::move( neis ), std::move( validGraphVerts ), std::move( ends ), std::move( validGraphEdges ) );
}

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
void MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::constructHeap_()
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

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
std::optional<Graph::EndVertices> MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::mergeNext_()
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

template<typename SegmentData, typename SegmentBdData, typename MergePenalty>
Expected<GroupOrder> MeshSegmenter<SegmentData, SegmentBdData, MergePenalty>::run_( const ProgressCallback& progress )
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

/// simple data of one segment for MeshSegmenter: its area
struct SimpleSegmentData
{
    double area = 0;
    SimpleSegmentData& operator +=( const SimpleSegmentData& r ) { area += r.area; return * this; }
};

/// simple data of the common boundary of two segments for MeshSegmenter
struct SimpleSegmentBdData
{
    /// the length of the boundary
    double length = 0;
    /// the integral of some curvature metric along the boundary
    double lengthCurv = 0;
    SimpleSegmentBdData& operator +=( const SimpleSegmentBdData& r ) { length += r.length; lengthCurv += r.lengthCurv; return * this; }
};

/// simple merge penalty for MeshSegmenter, which prefers merging
/// smaller by area segments, with long common boundary, and with low average curvature metric on it
struct SimpleMergePenalty
{
    double operator()( const SimpleSegmentData& a, const SimpleSegmentData& b, const SimpleSegmentBdData& bd ) const
    {
        if ( bd.length <= 0 )
            return prohibitMergePenalty;
        return std::min( a.area, b.area ) * bd.lengthCurv / ( bd.length * bd.length );
    }
};

} //namespace MR
