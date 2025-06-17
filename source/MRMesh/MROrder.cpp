#include "MROrder.h"
#include "MRBox.h"
#include "MRBuffer.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <span>

namespace MR
{

namespace
{

struct FacePoint
{
    FacePoint( NoInit ) noexcept : pt( noInit ), f( noInit ) {}

    Vector3f pt; // minimal bounding box point of the face
    FaceId f;
};
static_assert( sizeof( FacePoint ) == 16 );

using FacePointSpan = std::span<NoDefInit<FacePoint>>;

// [0, result) will go to left span and [result, lastLeaf) - to the right child
size_t partitionFacePoints( const FacePointSpan & span )
{
    assert( span.begin() + 1 < span.end() );
    Box3f box;
    for ( const auto & fp : span )
        box.include( fp.pt );

    // define total order of face points: no two distinct face points must be equivalent
    auto less = [sortedDims = findSortedBoxDims( box )]( const FacePoint & a, const FacePoint & b )
    {
        // first compare (and split later) by the largest box dimension
        for ( int i = decltype( sortedDims )::elements - 1; i >= 0; --i )
        {
            const int splitDim = sortedDims[i];
            const auto aDim = a.pt[splitDim];
            const auto bDim = b.pt[splitDim];
            if ( aDim != bDim )
                return aDim < bDim;
        }
        // if two face points have equal coordinates then compare by id to distinguish them
        return a.f < b.f;
    };

    const auto mid = span.size() / 2;
    std::nth_element( span.begin(), span.begin() + mid, span.end(), less );
    return mid;
}

// optimally orders given span, optionally splitting the job on given number of threads
void orderFacePoints( const FacePointSpan & span, int numThreads )
{
    if ( numThreads >= 2 && span.size() >= 32 )
    {
        // split the span between two threads
        const auto mid = partitionFacePoints( span );
        const int rThreads = numThreads / 2;
        const int lThreads = numThreads - rThreads;
        tbb::task_group group;
        group.run( [&] () { orderFacePoints( FacePointSpan( span.data() + mid, span.size() - mid ), rThreads ); } );
        orderFacePoints( FacePointSpan( span.data(), mid ), lThreads );
        group.wait();
        return;
    }

    // process the span in this thread only
    Timer t( "finishing" );
    std::vector<FacePointSpan> stack;
    stack.push_back( span );

    while ( !stack.empty() )
    {
        const auto x = stack.back();
        stack.pop_back();
        const auto mid = partitionFacePoints( x );
        if ( mid + 1 < x.size() )
            stack.emplace_back( x.data() + mid, x.size() - mid );
        if ( mid > 1 )
            stack.emplace_back( x.data(), mid );
    }
}

} // anonymous namespace

FaceBMap getOptimalFaceOrdering( const Mesh & mesh )
{
    MR_TIMER;

    FaceBMap res;
    const auto numFaces = mesh.topology.numValidFaces();

    res.b.resize( mesh.topology.faceSize() );
    res.tsize = numFaces;

    Buffer<FacePoint, FaceId> facePoints( numFaces );
    const bool packed = numFaces == mesh.topology.faceSize();
    if ( !packed )
    {
        FaceId n = 0_f;
        for ( FaceId f = 0_f; f < res.b.size(); ++f )
            if ( mesh.topology.hasFace( f ) )
                facePoints[n++].f = f;
            else
                res.b[f] = FaceId{};
    }

    // compute minimal point of each face
    tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, facePoints.endId() ),
        [&]( const tbb::blocked_range<FaceId>& range )
    {
        for ( FaceId i = range.begin(); i < range.end(); ++i )
        {
            FaceId f;
            if ( packed )
                facePoints[i].f = f = FaceId( i );
            else
                f = facePoints[i].f;
            Box3f box;
            Vector3f a, b, c;
            mesh.getTriPoints( f, a, b, c );
            box.include( a );
            box.include( b );
            box.include( c );
            facePoints[i].pt = box.min;
        }
    } );

    if ( facePoints.size() > 1 )
    {
        // to equally balance the load on threads, subdivide the task on
        // a power of two subtasks, which is at least twice the hardware concurrency
        int numThreads = 1;
        int target = (int)tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism );
        if ( target > 1 )
            numThreads *= 2;
        while ( target > 1 )
        {
            numThreads *= 2;
            target = ( target + 1 ) / 2;
        }
        orderFacePoints( { begin( facePoints ), facePoints.size() }, numThreads );
    }

    tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, facePoints.endId() ),
        [&]( const tbb::blocked_range<FaceId>& range )
    {
        for ( FaceId newf = range.begin(); newf < range.end(); ++newf )
        {
            res.b[facePoints[newf].f] = newf;
        }
    } );
    return res;
}

VertBMap getVertexOrdering( const FaceBMap & faceMap, const MeshTopology & topology )
{
    MR_TIMER;

    struct OrderedVertex
    {
        OrderedVertex( NoInit ) : v( noInit ) {}
        OrderedVertex( VertId v, std::uint32_t f ) noexcept : v( v ), f( f ) {}
        VertId v;
        std::uint32_t f; // the smallest nearby face
        bool operator <( const OrderedVertex & b ) const
            { return std::tie( f, v ) < std::tie( b.f, b.v ); } // order vertices by f
    };
    static_assert( sizeof( OrderedVertex ) == 8 );
    /// mapping: new vertex id -> old vertex id in v-field
    using VertexOrdering = Buffer<OrderedVertex, VertId>;

    assert( topology.lastValidFace() < (int)faceMap.b.size() );
    VertexOrdering ord( topology.vertSize() );

    Timer t( "fill" );
    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( !topology.hasVert( v ) )
            {
                // put at the very end after sorting
                ord[v] = OrderedVertex{ v, ~std::uint32_t(0) };
                continue;
            }
             // if no incident faces, then put after other valid vertices but before invalid vertices
            auto f = std::uint32_t( -(int)v - 2 );
            for ( EdgeId e : orgRing( topology, v ) )
                f = std::min( f, std::uint32_t( getAt( faceMap.b, topology.left( e ) ) ) );
            ord[v] = OrderedVertex{ v, f };
        }
    } );

    t.restart( "sort" );
    tbb::parallel_sort( ord.data(), ord.data() + ord.size() );

    VertBMap res;
    res.b.resize( topology.vertSize() );
    res.tsize = topology.numValidVerts();
    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            res.b[ord[v].v] = v < res.tsize ? v : VertId{};
        }
    } );

    return res;
}

UndirectedEdgeBMap getEdgeOrdering( const FaceBMap & faceMap, const MeshTopology & topology )
{
    MR_TIMER;

    struct OrderedEdge
    {
        OrderedEdge( NoInit ) noexcept : ue( noInit ) {}
        OrderedEdge( UndirectedEdgeId ue, std::uint32_t f ) noexcept : ue( ue ), f( f ) {}
        UndirectedEdgeId ue;
        std::uint32_t f; // the smallest nearby face
        bool operator <( const OrderedEdge & b ) const
            { return std::tie( f, ue ) < std::tie( b.f, b.ue ); } // order vertices by f
    };
    static_assert( sizeof( OrderedEdge ) == 8 );
    /// mapping: new vertex id -> old vertex id in v-field
    using EdgeOrdering = Buffer<OrderedEdge, UndirectedEdgeId>;

    assert( topology.lastValidFace() < (int)faceMap.b.size() );
    EdgeOrdering ord( topology.undirectedEdgeSize() );

    Timer t( "fill" );
    std::atomic<int> notLoneEdges{0};
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ),
    [&]( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        int myNotLoneEdges = 0;
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            const auto l = topology.left( ue );
            const auto r = topology.right( ue );
            if ( !l && !r )
            {
                if ( topology.isLoneEdge( ue ) )
                {
                    // put at the very end after sorting
                    ord[ue] = OrderedEdge{ ue, std::uint32_t( -1 ) };
                }
                else
                {
                    // put after edges with valid left/right faces but before lone edges after sorting
                    ord[ue] = OrderedEdge{ ue, std::uint32_t( -(int)ue - 2 ) };
                    ++myNotLoneEdges;
                }
            }
            else
            {
                auto f = std::min(
                    std::uint32_t( getAt( faceMap.b, l ) ),
                    std::uint32_t( getAt( faceMap.b, r ) ) );
                assert ( int(f) >= 0 );
                ord[ue] = OrderedEdge{ ue, f };
                ++myNotLoneEdges;
            }
        }
        notLoneEdges.fetch_add( myNotLoneEdges, std::memory_order_relaxed );
    } );

    t.restart( "sort" );
    tbb::parallel_sort( ord.data(), ord.data() + ord.size() );

    UndirectedEdgeBMap res;
    res.b.resize( topology.undirectedEdgeSize() );
    res.tsize = notLoneEdges;
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ),
    [&]( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            res.b[ord[ue].ue] = ue < res.tsize ? ue : UndirectedEdgeId{};
        }
    } );

    return res;
}

} //namespace MR
