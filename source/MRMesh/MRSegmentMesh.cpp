#include "MRSegmentMesh.h"
#include "MRMesh.h"
#include "MRGraph.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRHeap.h"
#include "MRUnionFind.h"
#include "MRColor.h"
#include "MRTimer.h"
#include <cfloat>
#include <optional>

namespace MR
{

namespace
{

static constexpr double ProhibitMergePenalty = DBL_MAX;

class MeshSegmenter
{
public:
    static Expected<GroupOrder> run( const Mesh& mesh, const EdgeMetric& curvMetric, const ProgressCallback& progress );

private:
    MeshSegmenter( const Mesh& mesh, const EdgeMetric& curvMetric ) : mesh_( mesh ), curvMetric_( curvMetric ) {}
    Expected<GroupOrder> run_( const ProgressCallback& progress );

    double mergePenalty_( Graph::EdgeId ge ) const;

    void constructGraph_();
    void constructHeap_();
    std::optional<Graph::EndVertices> mergeNext_();

private:
    const Mesh& mesh_;
    const EdgeMetric& curvMetric_;

    Graph graph_;

    struct SegmentData
    {
        double area = 0;
        SegmentData& operator +=( const SegmentData& r ) { area += r.area; return * this; }
    };
    Vector<SegmentData, Graph::VertId> graphVertData_;

    struct SegmentBdData
    {
        double length = 0;
        double lengthCurv = 0;
        SegmentBdData& operator +=( const SegmentBdData& r ) { length += r.length; lengthCurv += r.lengthCurv; return * this; }
    };
    Vector<SegmentBdData, Graph::EdgeId> graphEdgeData_;

    using Heap = MR::Heap<double, GraphEdgeId, std::greater<double>>;
    Heap heap_;
};

Expected<GroupOrder> MeshSegmenter::run( const Mesh& mesh, const EdgeMetric& curvMetric, const ProgressCallback& progress )
{
    MR_TIMER;

    MeshSegmenter s( mesh, curvMetric );

    s.constructGraph_();
    if ( !reportProgress( progress, 0.1f ) )
        return unexpectedOperationCanceled();

    s.constructHeap_();
    if ( !reportProgress( progress, 0.2f ) )
        return unexpectedOperationCanceled();

    return s.run_( subprogress( progress, 0.2f, 1.0f ) );
}

inline double MeshSegmenter::mergePenalty_( Graph::EdgeId ge ) const
{
    const auto & ed = graphEdgeData_[ge];
    if ( ed.length <= 0 )
        return ProhibitMergePenalty;
    const auto & vv = graph_.ends( ge );
    return std::min( graphVertData_[vv.v0].area, graphVertData_[vv.v1].area ) * ed.lengthCurv / sqr( ed.length );
}

void MeshSegmenter::constructGraph_()
{
    MR_TIMER;

    // initially:
    //   Graph::VertId = Mesh::FaceId
    //   Graph::EdgeId = Mesh::UndirectedEdgeId

    graphEdgeData_.resize( mesh_.topology.undirectedEdgeSize() );
    Graph::EndsPerEdge ends( mesh_.topology.undirectedEdgeSize() );
    Graph::EdgeBitSet validGraphEdges;
    static_cast<BitSet&>( validGraphEdges ) = mesh_.topology.findNotLoneUndirectedEdges();
    BitSetParallelFor( validGraphEdges, [&]( Graph::EdgeId ge )
    {
        UndirectedEdgeId ue( (int)ge );

        Graph::EndVertices vv;
        vv.v0 = Graph::VertId( (int)mesh_.topology.left( ue ) );
        vv.v1 = Graph::VertId( (int)mesh_.topology.right( ue ) );
        if ( !vv.v0 || !vv.v1 )
        {
            validGraphEdges.reset( ge );
            return;
        }

        assert( vv.v0 != vv.v1 );
        if ( vv.v1 < vv.v0 )
            std::swap( vv.v0, vv.v1 );
        ends[ge] = vv;

        const auto len = mesh_.edgeLength( ue );
        graphEdgeData_[ge] =
        {
            .length = len,
            .lengthCurv = len * curvMetric_( ue )
        };
    } );

    graphVertData_.resize( mesh_.topology.faceSize() );
    Graph::NeighboursPerVertex neis( mesh_.topology.faceSize() );
    Graph::VertBitSet validGraphVerts;
    static_cast<BitSet&>( validGraphVerts ) = mesh_.topology.getValidFaces();
    BitSetParallelFor( validGraphVerts, [&]( Graph::VertId gv )
    {
        FaceId f( (int)gv );
        Graph::Neighbours n;
        n.reserve( mesh_.topology.getFaceDegree( f ) );
        for ( auto e : leftRing( mesh_.topology, f ) )
        {
            Graph::EdgeId ge( int( e.undirected() ) );
            assert( mesh_.topology.left( e ) = f );
            if ( mesh_.topology.right( e ) )
                n.push_back( ge );
        }
        graphVertData_[gv].area = mesh_.area( f );
        std::sort( n.begin(), n.end() );
        neis[gv] = std::move( n );
    } );
    graph_.construct( std::move( neis ), std::move( validGraphVerts ), std::move( ends ), std::move( validGraphEdges ) );
}

void MeshSegmenter::constructHeap_()
{
    MR_TIMER;

    std::vector<Heap::Element> elements( mesh_.topology.undirectedEdgeSize(), { .val = ProhibitMergePenalty } );
    ParallelFor( graphEdgeData_, [&]( Graph::EdgeId ge )
    {
        elements[ge].id = ge;
        if ( !graph_.valid( ge ) )
            return;
        elements[ge].val = mergePenalty_( ge );
    } );
    heap_ = Heap( std::move( elements ) );
}

std::optional<Graph::EndVertices> MeshSegmenter::mergeNext_()
{
    const auto [ge, penalty] = heap_.top();
    if ( penalty == ProhibitMergePenalty )
        return std::nullopt;
    assert( graph_.valid( ge ) );
    assert( penalty == mergePenalty_( ge ) );
    heap_.setSmallerValue( ge, ProhibitMergePenalty );

    auto ends = graph_.ends( ge );
    graphVertData_[ends.v0] += graphVertData_[ends.v1];
    graph_.merge( ends.v0, ends.v1, [&]( Graph::EdgeId remnant, Graph::EdgeId dead )
        {
            graphEdgeData_[remnant] += graphEdgeData_[dead];
            heap_.setSmallerValue( dead, ProhibitMergePenalty );
        } );

    for ( auto ne : graph_.neighbours( ends.v0 ) )
        heap_.setValue( ne, mergePenalty_( ne ) );

    return ends;
}

Expected<GroupOrder> MeshSegmenter::run_( const ProgressCallback& progress )
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

/// all colors here are on cube's boundary intersected by a skew plane, which makes a hexagon
struct HexPalette
{
    /// different colors
    std::vector<Color> colors;

    /// recommended step from previous color to next color, to have big visual difference, and visit all colors in long run
    static constexpr int STEP = 17;

    HexPalette();
};

HexPalette::HexPalette()
{
    static constexpr int CORNER_COLORS = 6;
    static constexpr int SIDE_COLORS = 5; // num colors between two corner colors + 1
    // for any color c: dot( c, [1,1,1] ) = 1
    static const Vector3f cornerColors[CORNER_COLORS + 1] =
    {
        { 1.0, 0.0, 0.0 },
        { 0.5, 0.5, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.5, 0.5 },
        { 0.0, 0.0, 1.0 },
        { 0.5, 0.0, 0.5 },
        { 1.0, 0.0, 0.0 }
    };
    colors.reserve( CORNER_COLORS * SIDE_COLORS );
    for ( int corner = 0; corner < CORNER_COLORS; ++corner )
    {
        for ( int i = 0; i < SIDE_COLORS; ++i )
        {
            auto v = lerp( cornerColors[corner], cornerColors[corner+1], float(i) / SIDE_COLORS );
            colors.emplace_back( v );
        }
    }
    assert( colors.size() == CORNER_COLORS * SIDE_COLORS );
}

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const Mesh& mesh, const EdgeMetric& curvMetric, const ProgressCallback& progress )
{
    MR_TIMER;
    if ( !curvMetric )
        return unexpected( "no curvMetric given" );

    return MeshSegmenter::run( mesh, curvMetric, progress );
}

UndirectedEdgeBitSet findSegmentBoundaries( const MeshTopology& topology,
    const GroupOrder& groupOrder, int numSegments, FaceColors* outFaceColors )
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

    if ( outFaceColors )
    {
        outFaceColors->resizeNoInit( topology.faceSize() );
        HashMap<FaceId, Color> root2Color;
        HexPalette palette;

        // give colors to segments ignoring the contrast on their boundaries
        int nextColor = 0;
        for ( auto f : topology.getValidFaces() )
        {
            if ( roots[f] != f )
                continue;
            root2Color[f] = palette.colors[nextColor];
            nextColor = ( nextColor + HexPalette::STEP ) % palette.colors.size();
        }
        BitSetParallelForAll( topology.getValidFaces(), [&]( FaceId& f )
        {
            auto it = root2Color.find( roots[f] );
            assert( it != root2Color.end() );
            (*outFaceColors)[f] = it->second;
        } );
    }

    return res;
}

} //namespace MR
