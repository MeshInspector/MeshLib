#include "MRAABBTree.h"
#include "MRSimd4.hpp"
#include "MRInplaceStack.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRPrefetch.h"
#include "MRRegionBoundary.h"
#include "MRAABBTreeCenters.hpp"
#include "MRTriBox.hpp"
#include "MRTriMesh.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <tbb/parallel_invoke.h>

namespace MR
{

namespace
{

/// the box of a triangle of AABBTree, micro expanded to avoid leaks due to float errors
Box3f wideTriBox( const VertCoords & points, const AABBTree::LeafTri & lt )
{
    return insignificantlyExpandedTriBox( points[lt.verts[0]], points[lt.verts[1]], points[lt.verts[2]] );
}

/// where the processor cannot read ahead (WebAssembly has no prefetch), the build records keep the coordinates of the triangles,
/// so that each triangle of a leaf is one random read instead of four (its record and three points)
#if defined(__EMSCRIPTEN__) || defined(MR_AABBTREE_FACE_COORDS)
constexpr bool cFaceCoords = true;
#else
constexpr bool cFaceCoords = false;
#endif

/// a triangle of the build with the coordinates of its vertices
struct FaceTriCoords
{
    FaceId face;
    ThreeVertIds verts;
    Vector3f p[3];

    explicit FaceTriCoords( NoInit ) noexcept : face( noInit ), verts{ VertId( noInit ), VertId( noInit ), VertId( noInit ) },
        p{ Vector3f( noInit ), Vector3f( noInit ), Vector3f( noInit ) } { }
};

/// the record of a triangle in the build
using FaceTri = std::conditional_t<cFaceCoords, FaceTriCoords, AABBTree::LeafTri>;

/// the coordinates of the vertices of triangle ft: from the points, or from the record if it keeps them
inline void triPoints( const AABBTree::LeafTri & ft, const VertCoords & points, Vector3f ( &p )[3] )
{
    for ( int k = 0; k < 3; ++k )
        p[k] = points[ft.verts[k]];
}
inline void triPoints( const FaceTriCoords & ft, const VertCoords &, Vector3f ( &p )[3] )
{
    for ( int k = 0; k < 3; ++k )
        p[k] = ft.p[k];
}

/// keeps the coordinates of the vertices in the record if it has them
inline void keepPoints( AABBTree::LeafTri &, const Vector3f ( & )[3] ) { }
inline void keepPoints( FaceTriCoords & ft, const Vector3f ( &p )[3] )
{
    for ( int k = 0; k < 3; ++k )
        ft.p[k] = p[k];
}

/// the numFaces triangles (faces and vertices), the doubled centers of their boxes, and the box of the centers;
/// the leaves read the triangles in random order, so they are kept small (no boxes) to stay in the cache together with the points,
/// from which the leaves take the coordinates and the boxes
Buffer<FaceTri> makeFaceTris( const MeshPart & mp, int numFaces, Buffer<TreeCenter> & centers, Box3f & centersBox )
{
    Buffer<FaceTri> faceTris( numFaces );
    centers = Buffer<TreeCenter>( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            faceTris[n++].face = f;
    }

    const auto & topology = mp.mesh.topology;
    const auto & points = mp.mesh.points;
    const auto faceOf = [&] ( int i ) { return packed ? FaceId( i ) : faceTris[i].face; };
    // the records of a triangle are often far from the previous triangle's in memory, so they are read ahead:
    // its first edge's 2 * cAhead triangles ahead, its next edge's cAhead triangles ahead, and its points cAhead / 4 triangles ahead
    constexpr int cAhead = 16;
    constexpr int cBlock = 1024; // the centers are included in their box by blocks, while in the cache
    centersBox = tbb::parallel_reduce( tbb::blocked_range<int>( 0, numFaces ), Box3f(),
        [&] ( const tbb::blocked_range<int> & range, Box3f box )
    {
        const int end = range.end();
        for ( int b = range.begin(); b < end; b += cBlock )
        {
            for ( int i = b, be = std::min( end, b + cBlock ); i < be; ++i )
            {
                if ( i + 2 * cAhead < end )
                    topology.prefetchEdge( topology.edgeWithLeft( faceOf( i + 2 * cAhead ) ) );
                if ( i + cAhead < end )
                    topology.prefetchEdge( topology.prev( topology.edgeWithLeft( faceOf( i + cAhead ) ).sym() ) );
                if ( i + cAhead / 4 < end )
                    for ( auto v : topology.getTriVerts( faceOf( i + cAhead / 4 ) ) )
                        MR_PREFETCH( &points[v] );
                auto & ft = faceTris[i];
                const FaceId f = faceOf( i );
                ft.face = f;
                ft.verts = topology.getTriVerts( f );
                const Vector3f p[3] = { points[ft.verts[0]], points[ft.verts[1]], points[ft.verts[2]] };
                keepPoints( ft, p );
                Box3f triBox( noInit );
                insignificantlyExpandedTriBoxAndCenter( p[0], p[1], p[2], i, triBox, centers[i] );
            }
            box = includeCenters( box, centers.data() + b, std::min( end, b + cBlock ) - b );
        }
        return box;
    },
        [] ( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    return faceTris;
}

/// the triangles of tmesh (all if region is nullptr), see makeFaceTris above
Buffer<FaceTri> makeFaceTris( const TriMesh & tmesh, const FaceBitSet * region, int numFaces,
    Buffer<TreeCenter> & centers, Box3f & centersBox )
{
    const auto & points = tmesh.points;
    const auto & t = tmesh.tris;
    Buffer<FaceTri> faceTris( numFaces );
    centers = Buffer<TreeCenter>( numFaces );
    const bool packed = numFaces == (int)t.size();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : *region )
            faceTris[n++].face = f;
    }

    const auto faceOf = [&] ( int i ) { return packed ? FaceId( i ) : faceTris[i].face; };
    constexpr int cAhead = 4; // the points of a triangle are read cAhead triangles ahead
    constexpr int cBlock = 1024; // the centers are included in their box by blocks, while in the cache
    centersBox = tbb::parallel_reduce( tbb::blocked_range<int>( 0, numFaces ), Box3f(),
        [&] ( const tbb::blocked_range<int> & range, Box3f box )
    {
        const int end = range.end();
        for ( int b = range.begin(); b < end; b += cBlock )
        {
            for ( int i = b, be = std::min( end, b + cBlock ); i < be; ++i )
            {
                if ( i + cAhead < end )
                    for ( auto v : t[faceOf( i + cAhead )] )
                        MR_PREFETCH( &points[v] );
                const FaceId f = faceOf( i );
                const ThreeVertIds v = t[f];
                // not ft = { f, v }, which MSVC assembles on the stack and copies by one wider load, waiting for the stores
                auto & ft = faceTris[i];
                ft.face = f;
                ft.verts = v;
                const Vector3f p[3] = { points[v[0]], points[v[1]], points[v[2]] };
                keepPoints( ft, p );
                Box3f triBox( noInit );
                insignificantlyExpandedTriBoxAndCenter( p[0], p[1], p[2], i, triBox, centers[i] );
            }
            box = includeCenters( box, centers.data() + b, std::min( end, b + cBlock ) - b );
        }
        return box;
    },
        [] ( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    return faceTris;
}

/// splits each AABBTree node in two binary levels, each at the median of the triangle box centers
/// along the largest extent of these centers; so the nodes at binary depth d have ( numTris >> d ) or ( numTris >> d ) + 1 triangles
class AABBTreeWideMaker
{
public:
    using SimdNode = AABBTree::SimdNode;
    using SimdLeaf = AABBTree::SimdLeaf;
    using SimdBoxes = AABBTree::SimdBoxes;
    using LeafTri = AABBTree::LeafTri;

    /// counts the nodes of the tree of numTris triangles
    explicit AABBTreeWideMaker( int numTris ) : numTris_( numTris )
    {
        for ( auto & c : cnt_ )
            c = { Cnt{ -1, -1 }, Cnt{ -1, -1 } };
        total_ = count_( 0, numTris_ );
    }

    [[nodiscard]] int numInternal() const { return total_.nodes - total_.leaves; }
    [[nodiscard]] int numLeaves() const { return total_.leaves; }

    /// fills the internal nodes, the leaves and the triangles in them (all of the sizes for numTris)
    void make( const Buffer<FaceTri> & faceTris, Buffer<TreeCenter> & centers, const Box3f & centersBox, const VertCoords & points,
        SimdNode * nodes, SimdLeaf * leaves, LeafTri * leafTris )
    {
        assert( (int)faceTris.size() == numTris_ && (int)centers.size() == numTris_ );
        faceTris_ = faceTris.data();
        centers_ = centers.data();
        points_ = &points;
        nodes_ = nodes;
        leaves_ = leaves;
        leafTris_ = leafTris;
        streamLeafTris_ = reinterpret_cast<std::uintptr_t>( leafTris ) % 16 == 0;
        parallelSplits_ = numTris_ >= cParallelSplitNum
            && tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) > 1
            && tbb::this_task_arena::max_concurrency() > 1;
        if ( numTris_ >= cBucketSplitNum )
            scratch_ = Buffer<TreeCenter>( numTris_ );
        const Child root{ 0, { 0, numTris_, false, true, centersBox } };
        if ( numTris_ <= AABBTree::MaxLeafTris )
            make_( ~0, 0, 0, root );
        else
            make_( 0, 1, 0, root );
        Simd4::streamFence();
    }

private:
    /// centers [first, first + num) either in centers_ or at the same positions in scratch_, and the box of them if known
    struct Range
    {
        int first, num;
        bool inScratch;
        bool hasBox = false;
        Box3f box;
    };
    struct Child
    {
        int depth;
        Range r;
    };
    /// the number of all nodes and of the leaves in a subtree
    struct Cnt
    {
        int nodes, leaves;
    };

    TreeCenter * data_( const Range & r ) const { return ( r.inScratch ? scratch_.data() : centers_ ) + r.first; }

    // the numbers of nodes in the subtree of a node at binary depth d with num triangles
    Cnt count_( int d, int num )
    {
        Cnt & res = cnt_[d][num - ( numTris_ >> d )];
        if ( res.nodes >= 0 )
            return res;
        if ( num <= AABBTree::MaxLeafTris )
            return res = { 1, 1 };
        Cnt r{ 1, 0 };
        auto add = [&r]( Cnt c ) { r.nodes += c.nodes; r.leaves += c.leaves; };
        const int halves[2] = { num / 2, num - num / 2 };
        for ( int h : halves )
        {
            if ( h <= AABBTree::MaxLeafTris )
                add( count_( d + 1, h ) );
            else
            {
                add( count_( d + 2, h / 2 ) );
                add( count_( d + 2, h - h / 2 ) );
            }
        }
        return res = r;
    }
    Cnt cntOf_( const Child & c ) const { return cnt_[c.depth][c.r.num - ( numTris_ >> c.depth )]; }

    Box3f centersBox_( const Range & r ) const
    {
        const TreeCenter * d = data_( r );
        if ( !parallelSplits_ || r.num < cParallelSplitNum )
            return includeCenters( Box3f(), d, r.num );
        return tbb::parallel_reduce( tbb::blocked_range<int>( 0, r.num, 1 << 14 ), Box3f(),
            [d]( const tbb::blocked_range<int> & br, Box3f b ) { return includeCenters( b, d + br.begin(), (int)br.size() ); },
            []( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    }

    // [first, first + num / 2) get the smaller centers along the largest extent of the centers (ties are ordered by index); returns the two halves
    std::array<Range, 2> split_( const Range & r )
    {
        const auto box = r.hasBox ? r.box : centersBox_( r );
        const auto sz = box.size();
        const int axis = sz.x >= sz.y ? ( sz.x >= sz.z ? 0 : 2 ) : ( sz.y >= sz.z ? 1 : 2 );
        TreeCenter * const data = data_( r );
        if ( r.num >= cBucketSplitNum )
        {
            const int chunkSize = parallelSplits_ && r.num >= cParallelSplitNum ? cChunk : r.num;
            Box3f halfBoxes[2];
            if ( splitCentersByBuckets( data, data_( { r.first, r.num, !r.inScratch, false, {} } ), r.num, axis, box.min[axis], box.max[axis], chunkSize, halfBoxes ) )
                return { Range{ r.first, r.num / 2, !r.inScratch, true, halfBoxes[0] }, Range{ r.first + r.num / 2, r.num - r.num / 2, !r.inScratch, true, halfBoxes[1] } };
        }
        selectCenter( data, data + r.num / 2, data + r.num, axis );
        return { Range{ r.first, r.num / 2, r.inScratch, false, {} }, Range{ r.first + r.num / 2, r.num - r.num / 2, r.inScratch, false, {} } };
    }

    // fills leaf l with the triangles of range r, returns the box of the leaf
    Box3f makeLeaf_( int l, const Range & r )
    {
        const int num = r.num;
        // the triangles of a leaf in the order of their indices
        static_assert( AABBTree::MaxLeafTris == 4, "the sorting network below is for up to 4 triangles" );
        TreeCenter * const leaf = data_( r );
        const auto cs = [leaf]( int x, int y ) { if ( leaf[y].i < leaf[x].i ) std::swap( leaf[x], leaf[y] ); };
        switch ( num )
        {
        case 2: cs( 0, 1 ); break;
        case 3: cs( 0, 1 ); cs( 1, 2 ); cs( 0, 1 ); break;
        case 4: cs( 0, 1 ); cs( 2, 3 ); cs( 0, 2 ); cs( 1, 3 ); cs( 1, 2 ); break;
        default: break;
        }
        SimdLeaf sl( noInit );
        LeafTri lts[AABBTree::MaxLeafTris]; // the unused slots stay invalid
        Box3f box;
        for ( int i = 0; i < AABBTree::MaxLeafTris; ++i )
        {
            auto & lt = lts[i];
            if ( i >= num )
            {
                sl.setEmpty( i );
                continue;
            }
            const auto & ft = faceTris_[leaf[i].i];
            Vector3f p[3];
            triPoints( ft, *points_, p );
            const auto triBox = insignificantlyExpandedTriBox( p[0], p[1], p[2] );
            box.include( triBox );
            lt.face = ft.face;
            lt.verts = ft.verts;
            sl.set( i, p[0], p[1], p[2] );
        }
        // the arrays are larger than caches, and every leaf is written once
        Simd4::streamCopy( leaves_[l], sl );
        static_assert( sizeof( lts ) % 16 == 0 );
        if ( streamLeafTris_ )
            Simd4::streamCopyBytes( leafTris_ + l * AABBTree::MaxLeafTris, lts, sizeof( lts ) );
        else
            std::memcpy( static_cast<void *>( leafTris_ + l * AABBTree::MaxLeafTris ), lts, sizeof( lts ) );
        return box;
    }

    // fills the subtree of c with its root at ref (as in SimdNode::child): the internal children of the root get the indices from iDesc,
    // the internal nodes below them follow, and the leaves of the subtree get the indices from lFirst in their order; returns the box of the subtree
    Box3f make_( int ref, int iDesc, int lFirst, Child c )
    {
        const int num = c.r.num;
        if ( num <= AABBTree::MaxLeafTris )
        {
            assert( ref == ~lFirst );
            return makeLeaf_( lFirst, c.r );
        }
        // the leaves read the triangles and their points in random order: a subtree of up to 64 triangles reads ahead
        // its triangles, and a subtree of up to 16 triangles (when they are in the cache) reads ahead their points
        if ( num <= 64 )
        {
            const TreeCenter * d = data_( c.r );
            for ( int i = 0; i < num; ++i )
            {
                if ( cFaceCoords || num > 16 )
                    MR_PREFETCH( faceTris_ + d[i].i );
                else
                    for ( auto v : faceTris_[d[i].i].verts )
                        MR_PREFETCH( &( *points_ )[v] );
            }
        }

        const auto halves = split_( c.r );
        std::array<Range, 2> quarters[2];
        const bool parallel = num >= 4096;
        if ( parallel )
            tbb::parallel_invoke( [&] { quarters[0] = split_( halves[0] ); }, [&] { quarters[1] = split_( halves[1] ); } );
        Child ch[AABBTree::MaxChildren];
        int k = 0;
        for ( int j = 0; j < 2; ++j )
        {
            if ( halves[j].num <= AABBTree::MaxLeafTris )
            {
                ch[k++] = { c.depth + 1, halves[j] };
                continue;
            }
            if ( !parallel )
                quarters[j] = split_( halves[j] );
            ch[k++] = { c.depth + 2, quarters[j][0] };
            ch[k++] = { c.depth + 2, quarters[j][1] };
        }

        // the internal children are contiguous, then the subtrees of them in the order of children; the leaves are in the order of the triangles
        int childRef[AABBTree::MaxChildren], childIDesc[AABBTree::MaxChildren], childLFirst[AABBTree::MaxChildren];
        int numInternalCh = 0;
        for ( int j = 0; j < k; ++j )
            if ( ch[j].r.num > AABBTree::MaxLeafTris )
                ++numInternalCh;
        int nextI = iDesc, nextDesc = iDesc + numInternalCh, nextL = lFirst;
        for ( int j = 0; j < k; ++j )
        {
            const auto cnt = cntOf_( ch[j] );
            childLFirst[j] = nextL;
            nextL += cnt.leaves;
            if ( ch[j].r.num > AABBTree::MaxLeafTris )
            {
                childRef[j] = nextI++;
                childIDesc[j] = nextDesc;
                nextDesc += cnt.nodes - cnt.leaves - 1;
            }
            else
            {
                childRef[j] = ~childLFirst[j];
                childIDesc[j] = 0;
            }
        }

        Box3f boxes[AABBTree::MaxChildren];
        if ( !parallel )
        {
            for ( int j = 0; j < k; ++j )
                boxes[j] = make_( childRef[j], childIDesc[j], childLFirst[j], ch[j] );
        }
        else
        {
            tbb::task_group group;
            for ( int j = 1; j < k; ++j )
                group.run( [&, j]
                {
                    boxes[j] = make_( childRef[j], childIDesc[j], childLFirst[j], ch[j] );
                    Simd4::streamFence();
                } );
            boxes[0] = make_( childRef[0], childIDesc[0], childLFirst[0], ch[0] );
            group.wait();
        }

        SimdNode sn( noInit );
        Box3f box;
        for ( int j = 0; j < AABBTree::MaxChildren; ++j )
        {
            if ( j < k )
            {
                sn.child[j] = childRef[j];
                sn.set( j, boxes[j] );
                box.include( boxes[j] );
            }
            else
            {
                sn.child[j] = 0;
                sn.setEmpty( j );
            }
        }
        Simd4::streamCopy( nodes_[ref], sn );
        return box;
    }

    const FaceTri * faceTris_ = nullptr;
    const VertCoords * points_ = nullptr;
    TreeCenter * centers_ = nullptr;
    Buffer<TreeCenter> scratch_; // the centers of a range move between centers_ and scratch_ by each split of the range by buckets
    static constexpr int cBucketSplitNum = 1024; // smaller ranges are split by selectCenter
    static constexpr int cParallelSplitNum = 1 << 17;
    static constexpr int cChunk = 1 << 14; // the centers counted and moved by one thread in a parallel split
    bool parallelSplits_ = false; // on one thread the parallel split only adds passes over the range
    SimdNode * nodes_ = nullptr;
    SimdLeaf * leaves_ = nullptr;
    LeafTri * leafTris_ = nullptr;
    bool streamLeafTris_ = false; // leafTris_ is 16-byte aligned for non-temporal stores
    int numTris_ = 0;
    Cnt total_;
    std::array<Cnt, 2> cnt_[64]; // by binary depth and the number of triangles minus ( numTris_ >> depth )
};

/// makes the tree of numTris triangles given by makeFaceTris( centers, centersBox )
template <typename MakeFaceTris>
void makeTree( int numTris, const VertCoords & points, const MakeFaceTris & makeFaceTris, std::vector<AABBTree::SimdNode, Allocator<AABBTree::SimdNode>> & nodes,
    std::vector<AABBTree::SimdLeaf, Allocator<AABBTree::SimdLeaf>> & leaves, Vector<AABBTree::LeafTri> & leafTris )
{
    if ( numTris <= 0 )
        return;
    AABBTreeWideMaker maker( numTris );
    Buffer<TreeCenter> centers;
    Box3f centersBox;
    const auto faceTris = makeFaceTris( centers, centersBox );
    // the pages are touched first by the parallel filling
    resizeNoInit( nodes, maker.numInternal() );
    resizeNoInit( leaves, maker.numLeaves() );
    leafTris.resizeNoInit( size_t( maker.numLeaves() ) * AABBTree::MaxLeafTris );
    maker.make( faceTris, centers, centersBox, points, nodes.data(), leaves.data(), leafTris.data() );
}

} // anonymous namespace

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;
    numTris_ = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    makeTree( numTris_, mp.mesh.points, [&] ( Buffer<TreeCenter> & centers, Box3f & centersBox ) { return makeFaceTris( mp, numTris_, centers, centersBox ); },
        simdNodes_, simdLeaves_, leafTris_ );
}

AABBTree::AABBTree( const TriMesh & tmesh, const FaceBitSet * region )
{
    MR_TIMER;
    numTris_ = region ? (int)region->count() : (int)tmesh.tris.size();
    makeTree( numTris_, tmesh.points, [&] ( Buffer<TreeCenter> & centers, Box3f & centersBox ) { return makeFaceTris( tmesh, region, numTris_, centers, centersBox ); },
        simdNodes_, simdLeaves_, leafTris_ );
}

AABBTree::SimdBoxes AABBTree::leafBoxes( int leaf ) const
{
    SimdBoxes res( noInit );
    insignificantlyExpandedTriBoxes4( simdLeaves_[leaf].coord, res.box );
    return res;
}

Box3f AABBTree::leafBox( int i ) const
{
    return insignificantlyExpandedTriBox( simdLeaves_[i / MaxLeafTris].coord, i % MaxLeafTris );
}

void AABBTree::setLeafTri_( int l, int i, const LeafTri & lt, const VertCoords & points )
{
    leafTris_[size_t( l ) * MaxLeafTris + i] = lt;
    setLeafTriGeometry_( l, i, lt.verts, points );
}

void AABBTree::setLeafTriGeometry_( int l, int i, const ThreeVertIds & verts, const VertCoords & points )
{
    simdLeaves_[l].set( i, points[verts[0]], points[verts[1]], points[verts[2]] );
}

Vector<Box3f, NodeId> AABBTree::getNodeBoxes() const
{
    MR_TIMER;
    Vector<Box3f, NodeId> res;
    if ( empty() )
        return res;
    res.resizeNoInit( numNodes() );
    res[NodeId( 0 )] = getBoundingBox();
    ParallelFor( size_t( 0 ), simdNodes_.size(), [&]( size_t n )
    {
        const auto & sn = simdNodes_[n];
        for ( int i = 0; i < MaxChildren; ++i )
            if ( sn.child[i] )
                res[nodeId( sn.child[i] )] = sn.get( i );
    } );
    return res;
}

std::vector<NodeId> AABBTree::getSubtrees( int minNum ) const
{
    MR_TIMER;
    assert( minNum > 0 );
    std::vector<NodeId> res;
    if ( empty() )
        return res;
    std::vector<int> refs{ rootRef() }, tmp;
    while ( refs.size() < minNum )
    {
        bool split = false;
        for ( int r : refs )
        {
            if ( r < 0 )
            {
                tmp.push_back( r );
                continue;
            }
            for ( int c : simdNodes_[r].child )
                if ( c )
                    tmp.push_back( c );
            split = true;
        }
        refs.swap( tmp );
        tmp.clear();
        if ( !split )
            break;
    }
    res.reserve( refs.size() );
    for ( int r : refs )
        res.push_back( nodeId( r ) );
    return res;
}

FaceBitSet AABBTree::getSubtreeFaces( NodeId subtreeRoot ) const
{
    MR_TIMER;
    FaceBitSet res;
    // every level adds at most MaxChildren - 1 nodes, and the tree has at most MaxDepth levels
    InplaceStack<int, 64> subtasks;
    subtasks.push( nodeRef( subtreeRoot ) );
    while ( !subtasks.empty() )
    {
        const int r = subtasks.top();
        subtasks.pop();
        if ( r < 0 )
        {
            for ( int i = 0; i < MaxLeafTris; ++i )
                if ( const auto f = leafTris_[size_t( ~r ) * MaxLeafTris + i].face )
                    res.autoResizeSet( f );
            continue;
        }
        const auto & sn = simdNodes_[r];
        for ( int i = MaxChildren - 1; i >= 0; --i )
            if ( sn.child[i] )
                subtasks.push( sn.child[i] );
    }
    return res;
}

NodeBitSet AABBTree::getNodesFromFaces( const FaceBitSet & faces ) const
{
    MR_TIMER;
    NodeBitSet res( numNodes() );
    if ( empty() )
        return res;
    const int numInternal = (int)simdNodes_.size();
    BitSetParallelForAll( res, [&]( NodeId n )
    {
        if ( int( n ) < numInternal )
            return;
        const auto * lt = leafTris_.data() + size_t( int( n ) - numInternal ) * MaxLeafTris;
        for ( int i = 0; i < MaxLeafTris && lt[i].face; ++i )
        {
            if ( faces.test( lt[i].face ) )
            {
                res.set( n );
                return;
            }
        }
    } );
    // children follow their parent
    for ( int n = numInternal - 1; n >= 0; --n )
    {
        for ( int c : simdNodes_[n].child )
        {
            if ( c && res.test( nodeId( c ) ) )
            {
                res.set( NodeId( n ) );
                break;
            }
        }
    }
    return res;
}

void AABBTree::getLeafOrder( FaceBMap & faceMap ) const
{
    MR_TIMER;
    int k = 0;
    for ( const auto & lt : leafTris_ )
        if ( lt.face )
            faceMap.b[lt.face] = FaceId( k++ );
    faceMap.tsize = decltype( faceMap.tsize )( k );
}

void AABBTree::renumber( const FaceBMap & faceMap, const VertBMap & vertMap )
{
    MR_TIMER;
    ParallelFor( leafTris_, [&]( size_t i )
    {
        auto & lt = leafTris_[i];
        if ( !lt.face )
            return;
        lt.face = faceMap.b[lt.face];
        for ( auto & v : lt.verts )
            v = vertMap.b[v];
    } );
}

void AABBTree::flipOrientation( const FaceBitSet * region )
{
    MR_TIMER;
    ParallelFor( size_t( 0 ), simdLeaves_.size(), [&]( size_t l )
    {
        auto & leaf = simdLeaves_[l];
        for ( int i = 0; i < MaxLeafTris; ++i )
        {
            auto & lt = leafTris_[l * MaxLeafTris + i];
            if ( !lt.face )
                break;
            if ( region && !region->test( lt.face ) )
                continue;
            std::swap( lt.verts[0], lt.verts[1] );
            for ( int c = 0; c < 3; ++c )
                std::swap( leaf.coord[0][c][i], leaf.coord[1][c][i] );
        }
    } );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( empty() )
        return;
    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );

    // the triangles with changed vertices in the leaves, and the boxes of the changed leaves
    const auto & points = mesh.points;
    BitSet changedLeaves;
    Buffer<Box3f> leafBox( simdLeaves_.size() );
    if ( !changedVerts )
    {
        // every leaf is assembled in registers and written by non-temporal stores, which do not read the old lines first
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, simdLeaves_.size(), 1024 ), [&]( const tbb::blocked_range<size_t> & range )
        {
            for ( size_t l = range.begin(); l < range.end(); ++l )
            {
                const auto * lts = leafTris_.data() + l * MaxLeafTris;
                SimdLeaf sl( noInit );
                Box3f box;
                for ( int i = 0; i < MaxLeafTris; ++i )
                {
                    if ( !lts[i].face )
                    {
                        sl.setEmpty( i );
                        continue;
                    }
                    const auto tb = wideTriBox( points, lts[i] );
                    box.include( tb );
                    sl.set( i, points[lts[i].verts[0]], points[lts[i].verts[1]], points[lts[i].verts[2]] );
                }
                Simd4::streamCopy( simdLeaves_[l], sl );
                leafBox[l] = box;
            }
            Simd4::streamFence();
        } );
    }
    else
    {
        changedLeaves.resize( simdLeaves_.size() );
        BitSetParallelForAll( changedLeaves, [&]( size_t l )
        {
            const auto * lts = leafTris_.data() + l * MaxLeafTris;
            bool changed = false;
            for ( int i = 0; i < MaxLeafTris && lts[i].face; ++i )
            {
                if ( !changedFaces.test( lts[i].face ) )
                    continue;
                setLeafTriGeometry_( int( l ), i, lts[i].verts, points );
                changed = true;
            }
            if ( !changed )
                return;
            changedLeaves.set( l );
            leafBox[l] = leafBoxes( int( l ) ).getUnion();
        } );
    }

    // the boxes of the changed children in the internal nodes, the subtrees of the top nodes in parallel;
    // returns whether the subtree of internal node n has changed
    auto refitNode = [&]( int n, int depth, auto && self ) -> bool
    {
        auto & sn = simdNodes_[n];
        bool changedCh[MaxChildren] = {};
        auto refitChild = [&]( int i )
        {
            const int c = sn.child[i];
            if ( c < 0 )
                changedCh[i] = !changedVerts || changedLeaves.test( ~c );
            else if ( c > 0 )
                changedCh[i] = self( c, depth + 1, self );
        };
        if ( depth < 3 )
            tbb::parallel_for( 0, MaxChildren, refitChild );
        else
            for ( int i = 0; i < MaxChildren; ++i )
                refitChild( i );
        bool changed = false;
        for ( int i = 0; i < MaxChildren; ++i )
        {
            if ( !changedCh[i] )
                continue;
            const int c = sn.child[i];
            sn.set( i, c < 0 ? leafBox[~c] : simdNodes_[c].getUnion() );
            changed = true;
        }
        return changed;
    };
    if ( !simdNodes_.empty() )
        refitNode( 0, 0, refitNode );
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !empty() );

    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        splitFaces.set( oldFace );

    // the leaf of every split face, and the new vertices and boxes of the split faces
    std::vector<FaceId> oldFaces; // in increasing order
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    std::vector<int> leafOfOld( oldFaces.size(), -1 );
    ParallelFor( size_t( 0 ), simdLeaves_.size(), [&]( size_t l )
    {
        for ( int i = 0; i < MaxLeafTris; ++i )
        {
            const auto face = leafTris_[l * MaxLeafTris + i].face;
            if ( !face )
                break;
            if ( !splitFaces.test( face ) )
                continue;
            const LeafTri lt( face, mesh.topology.getTriVerts( face ) );
            setLeafTri_( int( l ), i, lt, mesh.points );
            const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), face );
            assert( it != oldFaces.end() && *it == face );
            leafOfOld[it - oldFaces.begin()] = int( l );
        }
    } );

    // the new faces grouped by the leaf of their old face, in a deterministic order
    std::vector<std::pair<int, FaceId>> leafNew;
    leafNew.reserve( new2Old.size() );
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        const auto leaf = leafOfOld[std::lower_bound( oldFaces.begin(), oldFaces.end(), oldFace ) - oldFaces.begin()];
        if ( leaf < 0 )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        leafNew.emplace_back( leaf, newFace );
    }
    std::sort( leafNew.begin(), leafNew.end() );

    // the parent of every leaf with split faces: the internal node and the slot in it
    BitSet splitLeaves( simdLeaves_.size() );
    for ( const auto & ln : leafNew )
        splitLeaves.set( ln.first );
    std::vector<std::pair<int, int>> parentOf( simdLeaves_.size(), { -1, -1 } );
    ParallelFor( size_t( 0 ), simdNodes_.size(), [&]( size_t n )
    {
        const auto & sn = simdNodes_[n];
        for ( int i = 0; i < MaxChildren; ++i )
            if ( sn.child[i] < 0 && splitLeaves.test( ~sn.child[i] ) )
                parentOf[~sn.child[i]] = { int( n ), i };
    } );

    // appends a leaf with given faces, returns its reference
    auto addLeaf = [&]( const std::pair<int, FaceId> * faces, int num )
    {
        const int l = (int)simdLeaves_.size();
        simdLeaves_.emplace_back( noInit );
        leafTris_.resize( leafTris_.size() + MaxLeafTris );
        for ( int i = 0; i < MaxLeafTris; ++i )
        {
            if ( i < num )
            {
                const auto f = faces[i].second;
                const LeafTri lt( f, mesh.topology.getTriVerts( f ) );
                setLeafTri_( l, i, lt, mesh.points );
                continue;
            }
            simdLeaves_[l].setEmpty( i );
        }
        numTris_ += num;
        return ~l;
    };
    auto boxOf = [&]( int ref ) { return ref < 0 ? leafBoxes( ~ref ).getUnion() : simdNodes_[ref].getUnion(); };
    // appends a node with given children, more than MaxChildren of them are grouped in subtrees; the new nodes are appended,
    // so children have larger ids than their parents as in a constructed tree; returns the reference to the node and its box
    auto makeNode = [&]( const std::vector<int> & refs, auto && self ) -> std::pair<int, Box3f>
    {
        const int n = (int)refs.size();
        const int numCh = std::min( n, MaxChildren );
        const int node = (int)simdNodes_.size();
        simdNodes_.emplace_back( noInit );
        SimdNode sn( noInit );
        Box3f box;
        for ( int c = 0; c < MaxChildren; ++c )
        {
            if ( c >= numCh )
            {
                sn.child[c] = 0;
                sn.setEmpty( c );
                continue;
            }
            int ref;
            Box3f cb;
            if ( n <= MaxChildren )
            {
                ref = refs[c];
                cb = boxOf( ref );
            }
            else
            {
                std::vector<int> part( refs.begin() + c * n / numCh, refs.begin() + ( c + 1 ) * n / numCh );
                if ( part.size() == 1 )
                {
                    ref = part.front();
                    cb = boxOf( ref );
                }
                else
                    std::tie( ref, cb ) = self( part, self );
            }
            sn.child[c] = ref;
            sn.set( c, cb );
            box.include( cb );
        }
        simdNodes_[node] = sn;
        return { node, box };
    };

    // every changed leaf is replaced by a node with the old leaf as the first child and the leaves of the new faces
    bool grown = false; // whether the subtree of some split face is not inside the box of its old leaf
    for ( size_t k = 0; k < leafNew.size(); )
    {
        const int leaf = leafNew[k].first;
        std::vector<int> refs{ ~leaf }; // the old leaf keeps its triangles
        for ( ; k < leafNew.size() && leafNew[k].first == leaf; )
        {
            int num = 0;
            while ( num < MaxLeafTris && k + num < leafNew.size() && leafNew[k + num].first == leaf )
                ++num;
            refs.push_back( addLeaf( leafNew.data() + k, num ) );
            k += num;
        }
        const auto [p, slot] = parentOf[leaf];
        const auto [ref, box] = makeNode( refs, makeNode );
        if ( p < 0 )
        {
            // the old leaf was the root
            assert( ref == 0 );
            continue;
        }
        auto & pn = simdNodes_[p];
        grown = grown || !pn.get( slot ).contains( box );
        pn.child[slot] = ref;
        pn.set( slot, box );
    }

    // only a new vertex outside the box of its split face (e.g. due to rounding) makes the boxes of the ancestors to grow,
    // then update all internal nodes from the last to the root as in refit()
    if ( grown )
    {
        for ( int n = int( simdNodes_.size() ) - 1; n >= 0; --n )
        {
            auto & sn = simdNodes_[n];
            for ( int i = 0; i < MaxChildren; ++i )
                if ( sn.child[i] )
                    sn.set( i, boxOf( sn.child[i] ) );
        }
    }

    // every split leaf went one level deeper: the stacks of the traversals hold the nodes of MaxDepth levels only
    int maxDepth = 0;
    std::vector<std::pair<int, int>> stack{ { rootRef(), 0 } }; // reference and depth
    while ( !stack.empty() )
    {
        const auto [r, d] = stack.back();
        stack.pop_back();
        maxDepth = std::max( maxDepth, d );
        if ( r >= 0 )
            for ( int c : simdNodes_[r].child )
                if ( c )
                    stack.push_back( { c, d + 1 } );
    }
    if ( maxDepth > MaxDepth )
    {
        FaceBitSet faces( mesh.topology.faceSize() );
        for ( const auto & lt : leafTris_ )
            if ( lt.face )
                faces.set( lt.face );
        *this = AABBTree( MeshPart{ mesh, &faces } );
    }
}

} //namespace MR
