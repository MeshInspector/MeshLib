#pragma once

#include "MRAABBTree.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include "MRBuffer.h"
#include "MREnums.h"
#include "MRInplaceStack.h"
#include "MRSimd4.hpp"
#include "MRTriBox.hpp"
#include "MRVector.h"
#include <atomic>
#include <optional>
#include <concepts>
#include <utility>
#include <cstdint>

namespace MR
{

/// the same as tree.leafBoxes( leaf ), inlined in the traversals
inline AABBTree::SimdBoxes leafTriBoxes( const AABBTree & tree, int leaf )
{
    AABBTree::SimdBoxes res( noInit );
    insignificantlyExpandedTriBoxes4( tree.simdLeaves()[leaf].coord, res.box );
    return res;
}

/// the same as tree.leafBox( i ), inlined in the traversals
inline Box3f leafTriBox( const AABBTree & tree, int i )
{
    return insignificantlyExpandedTriBox( tree.simdLeaves()[i / AABBTree::MaxLeafTris].coord, i % AABBTree::MaxLeafTris );
}

/// a node of AABBTree in a traversal: its reference as in AABBTree::SimdNode::child and its box
struct WideNode
{
    int ref;
    Box3f box;
    [[nodiscard]] bool leaf() const { return ref < 0; }
};

/// the root of not-empty tree in a traversal
inline WideNode rootWideNode( const AABBTree & tree ) { return { tree.rootRef(), tree.getBoundingBox() }; }

/// i-th child of an internal node in a traversal
inline WideNode childWideNode( const AABBTree::SimdNode & sn, int i ) { return { sn.child[i], sn.get( i ) }; }

/// the number of the children of an internal node
inline int numChildren( const AABBTree::SimdNode & sn ) { return 1 + ( sn.child[1] != 0 ) + ( sn.child[2] != 0 ) + ( sn.child[3] != 0 ); }

/// a pair of nodes of two AABBTree (or of one tree) in their simultaneous traversal
struct WideNodePair
{
    WideNode a, b;
};

/// returns the bit mask of the four boxes (bit i for box i), which intersect given box (as Box3f::intersects) by one SIMD test
inline int simdBoxesIntersect( const AABBTree::SimdBoxes & sb, const Box3f & box )
{
    using namespace Simd4;
    M ok = andm( le( load( sb.box[0] ), set1( box.max.x ) ), le( set1( box.min.x ), load( sb.box[3] ) ) );
    ok = andm( ok, andm( le( load( sb.box[1] ), set1( box.max.y ) ), le( set1( box.min.y ), load( sb.box[4] ) ) ) );
    ok = andm( ok, andm( le( load( sb.box[2] ), set1( box.max.z ) ), le( set1( box.min.z ), load( sb.box[5] ) ) ) );
    return bits( ok ); // unused slots have empty boxes
}

/// intersect( aNode, bNode ) of splitWidePair together with
/// children( aSide, parent, other ) returning the bit mask of the children of parent (in tree a if aSide, otherwise in tree b),
/// for which intersect with node other of the other tree would return true
template<typename Pair, typename Children>
struct WideIntersect
{
    Pair pair;
    Children children;
    bool operator()( const WideNode & a, const WideNode & b ) const { return pair( a, b ); }
};
template<typename Pair, typename Children>
WideIntersect( Pair, Children ) -> WideIntersect<Pair, Children>;

/// pushes the pairs of each child of the bigger non-leaf node with the other node, if intersect( aNode, bNode ) of their boxes;
/// returns false and pushes nothing if both nodes are leaves
template<typename Intersect, typename Push>
bool splitWidePair( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, Intersect && intersect, Push && push )
{
    constexpr bool hasChildren = requires { intersect.children( true, s.a, s.b ); };
    if ( !s.a.leaf() && ( s.b.leaf() || s.a.box.volume() >= s.b.box.volume() ) )
    {
        const auto & sn = aTree.simdNodes()[s.a.ref];
        if constexpr ( hasChildren )
        {
            for ( int m = intersect.children( true, s.a, s.b ); m; m &= m - 1 )
                push( WideNodePair{ childWideNode( sn, Simd4::lowestBit( m ) ), s.b } );
        }
        else
        {
            for ( int i = 0; i < AABBTree::MaxChildren && sn.child[i]; ++i )
                if ( const auto c = childWideNode( sn, i ); intersect( c, s.b ) )
                    push( WideNodePair{ c, s.b } );
        }
        return true;
    }
    if ( !s.b.leaf() )
    {
        const auto & sn = bTree.simdNodes()[s.b.ref];
        if constexpr ( hasChildren )
        {
            for ( int m = intersect.children( false, s.b, s.a ); m; m &= m - 1 )
                push( WideNodePair{ s.a, childWideNode( sn, Simd4::lowestBit( m ) ) } );
        }
        else
        {
            for ( int i = 0; i < AABBTree::MaxChildren && sn.child[i]; ++i )
                if ( const auto c = childWideNode( sn, i ); intersect( s.a, c ) )
                    push( WideNodePair{ s.a, c } );
        }
        return true;
    }
    return false;
}

/// the same for the traversal of one tree with itself: a node paired with itself gives each child paired with itself and the pairs of different children
template<typename Intersect, typename Push>
bool splitSelfWidePair( const AABBTree & tree, const WideNodePair & s, Intersect && intersect, Push && push )
{
    if ( s.a.ref != s.b.ref )
        return splitWidePair( tree, tree, s, intersect, push );
    if ( s.a.leaf() )
        return false;
    const auto & sn = tree.simdNodes()[s.a.ref];
    const int count = numChildren( sn );
    for ( int i = 0; i < count; ++i )
    {
        const auto ci = childWideNode( sn, i );
        push( WideNodePair{ ci, ci } );
        if constexpr ( requires { intersect.children( true, s.a, s.b ); } )
        {
            // the children after ci, whose boxes intersect the box of ci
            for ( int m = intersect.children( true, s.a, ci ) & int( ~0u << ( i + 1 ) ); m; m &= m - 1 )
                push( WideNodePair{ ci, childWideNode( sn, Simd4::lowestBit( m ) ) } );
        }
        else
        {
            for ( int j = i + 1; j < count; ++j )
                if ( const auto cj = childWideNode( sn, j ); intersect( ci, cj ) )
                    push( WideNodePair{ ci, cj } );
        }
    }
    return true;
}

/// calls onTris( i, j ) for the triangles (slots in leafTris()) of leaves s.a and s.b, whose boxes intersect;
/// the boxes of the triangles of both leaves are computed once from their coordinates:
/// triNode( i, aBox, bNode ) tests the box aBox of triangle i of s.a with the box of leaf s.b,
/// and tris( i, aBox, bLeaf, bBoxes ) returns the bit mask of the triangles of leaf s.b, whose boxes intersect aBox,
/// where bBoxes() returns the boxes of the triangles of s.b in tree b, computed on its first call (so not at all if the boxes after a transformation are cached);
/// in the traversal of one tree with itself (Self), a leaf paired with itself gives each pair of its triangles once, i < j;
/// stops and returns false as soon as onTris returns false
template<bool Self = false, typename TriNode, typename Tris, typename OnTris>
bool forWideLeafTris( const AABBTree & aTree, const AABBTree & bTree, const WideNodePair & s, TriNode && triNode, Tris && tris, OnTris && onTris )
{
    assert( s.a.leaf() && s.b.leaf() );
    const bool self = Self && s.a.ref == s.b.ref;
    const int aFirst = ~s.a.ref * AABBTree::MaxLeafTris;
    const int bFirst = ~s.b.ref * AABBTree::MaxLeafTris;
    const int aNum = aTree.numLeafTris( ~s.a.ref );
    const int bMask = ( 1 << bTree.numLeafTris( ~s.b.ref ) ) - 1;
    const auto aBoxes = leafTriBoxes( aTree, ~s.a.ref );
    std::optional<AABBTree::SimdBoxes> bComputed;
    const auto bBoxes = [&]() -> const AABBTree::SimdBoxes &
    {
        if ( self )
            return aBoxes;
        if ( !bComputed )
            bComputed.emplace( leafTriBoxes( bTree, ~s.b.ref ) );
        return *bComputed;
    };
    for ( int k = 0; k < aNum; ++k )
    {
        const Box3f aBox = aBoxes.get( k );
        if ( !self && !triNode( aFirst + k, aBox, s.b ) )
            continue;
        int m = tris( aFirst + k, aBox, s.b, bBoxes ) & bMask;
        if ( self )
            m &= int( ~0u << ( k + 1 ) );
        for ( ; m; m &= m - 1 )
            if ( !onTris( aFirst + k, bFirst + Simd4::lowestBit( m ) ) )
                return false;
    }
    return true;
}

/// sequentially subdivides the traversal from the given pairs on at most given number of levels to get subtasks for parallel processing:
/// they shall be not too many for this subdivision not to take too long, and not too few for enough parallelism later;
/// split( s, push ) returns false for a pair of leaves, which then becomes a subtask
template<typename Split>
Vector<WideNodePair> makeWideSubtasks( Vector<WideNodePair> subtasks, int levels, Split && split )
{
    Vector<WideNodePair> next, leafTasks;
    for ( int i = 0; i < levels && !subtasks.empty(); ++i )
    {
        for ( const auto & s : subtasks )
            if ( !split( s, [&next]( const WideNodePair & c ) { next.push_back( c ); } ) )
                leafTasks.push_back( s );
        subtasks.swap( next );
        next.clear();
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );
    return subtasks;
}

/// the box after the transformation
inline Box3f transformedBoxes( const Box3f & box, const AffineXf3f & xf ) { return transformed( box, xf ); }

/// each of the four boxes after the transformation, the same as transformed( box, xf ) of it
inline AABBTree::SimdBoxes transformedBoxes( const AABBTree::SimdBoxes & sb, const AffineXf3f & xf )
{
    AABBTree::SimdBoxes res( noInit );
    for ( int i = 0; i < 4; ++i )
        res.set( i, transformed( sb.get( i ), xf ) );
    return res;
}

/// the boxes (Box3f) or the groups of four boxes (AABBTree::SimdBoxes) after a transformation,
/// each computed on the first request (from any thread) and kept for the next ones
template<typename I, typename T = Box3f>
class LazyTransformedBoxes
{
public:
    LazyTransformedBoxes( size_t size, const AffineXf3f & xf ) : xf_( xf ), boxes_( size ), state_( size ) { }

    /// returns box i, which is src before the transformation (src can be a function returning it, called only if box i is not ready)
    template<typename Src>
    T get( I i, const Src & src ) const
    {
        auto & state = state_[i];
        if ( state.load( std::memory_order_acquire ) == cReady )
            return boxes_[i];
        T res;
        if constexpr ( std::invocable<const Src &> )
            res = transformedBoxes( src(), xf_ );
        else
            res = transformedBoxes( src, xf_ );
        // only the thread marking the box first stores it; the others meanwhile use their own copies
        std::uint8_t expected = cEmpty;
        if ( state.compare_exchange_strong( expected, cWriting, std::memory_order_relaxed ) )
        {
            boxes_[i] = res;
            state.store( cReady, std::memory_order_release );
        }
        return res;
    }

private:
    static constexpr std::uint8_t cEmpty = 0, cWriting = 1, cReady = 2;
    const AffineXf3f & xf_;
    mutable Buffer<T, I> boxes_; // not initialized: an element is written before its state becomes ready
    mutable Vector<std::atomic<std::uint8_t>, I> state_;
};

/// a test of boxes from given functions: one for a box, test( box ) -> bool, and one for four boxes at once, test( simdBoxes ) -> bit mask of passed ones
/// (the functions are members and not bases: deriving from a lambda with tail padding is warned about by MSVC for its ABI)
template<typename F1, typename F2>
struct WideBoxTest
{
    F1 f1;
    F2 f2;

    template<typename... A> requires std::invocable<const F1 &, A...>
    decltype(auto) operator()( A &&... a ) const { return f1( std::forward<A>( a )... ); }

    template<typename... A> requires ( !std::invocable<const F1 &, A...> && std::invocable<const F2 &, A...> )
    decltype(auto) operator()( A &&... a ) const { return f2( std::forward<A>( a )... ); }
};
template<typename F1, typename F2>
WideBoxTest( F1, F2 ) -> WideBoxTest<F1, F2>;

/// the squared distances between box b and four boxes, computed exactly as Box3f::getDistanceSq( b ) of each of them; infinite for empty boxes
inline Simd4::F boxBoxDistSq( const AABBTree::SimdBoxes & sb, const Box3f & b )
{
    using namespace Simd4;
    F res = set1( 0.0f );
    for ( int c = 0; c < 3; ++c )
    {
        const F lo = max( load( sb.box[c] ), set1( b.min[c] ) );
        const F hi = min( load( sb.box[c + 3] ), set1( b.max[c] ) );
        const F d = sub( lo, hi );
        res = add( res, select( gt( lo, hi ), mul( d, d ), set1( 0.0f ) ) );
    }
    return res;
}

/// bounds of the boxes of the triangles of a leaf as in AABBTree::leafBoxes( leaf ): inner - the boxes of their vertices, contained in them,
/// and outer - the same expanded by two units in the last place, containing them; returns the bit mask of the used slots with the bounds
/// not suitable for the decision (a NaN or an infinite coordinate) in bad, and of all used slots
inline int leafBoxBounds( const AABBTree::SimdLeaf & leaf, AABBTree::SimdBoxes & inner, AABBTree::SimdBoxes & outer, int & bad )
{
    using namespace Simd4;
    const F k = set1( 2.4e-7f ), tiny = set1( 1e-37f );
    const F c00 = load( leaf.coord[0][0] ), c01 = load( leaf.coord[0][1] );
    const int used = bits( orm( ne( c00, c00 ) , ne( c01, c01 ) ) ) ^ 15;
    M nan = lt( k, k ); // all false
    for ( int c = 0; c < 3; ++c )
    {
        const F a = load( leaf.coord[0][c] ), b = load( leaf.coord[1][c] ), cc = load( leaf.coord[2][c] );
        const F lo = min( min( a, b ), cc ), hi = max( max( a, b ), cc );
        store( inner.box[c], lo );
        store( inner.box[c + 3], hi );
        const F olo = sub( sub( lo, mul( abs( lo ), k ) ), tiny );
        const F ohi = add( add( hi, mul( abs( hi ), k ) ), tiny );
        store( outer.box[c], olo );
        store( outer.box[c + 3], ohi );
        // NaN in a coordinate, or an infinite one making NaN of the expansion
        nan = orm( nan, orm( ne( olo, olo ), ne( ohi, ohi ) ) );
    }
    bad = bits( nan ) & used;
    return used;
}

/// visits depth-first the nodes with boxes passing boxOk, children in their order, and calls onTri( i )
/// for the triangles (slots in leafTris()) of the visited leaves with boxes passing boxOk;
/// boxOk( box ) tests the box of the root, and boxOk( simdBoxes ) the boxes of the children of a node or of the triangles of a leaf, see WideBoxTest;
/// boxOk must be monotone: a box containing a passing box passes too (as in the tests of intersection with a set);
/// stops as soon as onTri returns Processing::Stop
template<typename BoxOk, typename OnTri>
void forWideTris( const AABBTree & tree, BoxOk && boxOk, OnTri && onTri )
{
    if ( tree.empty() || !boxOk( tree.getBoundingBox() ) )
        return;
    // every level adds at most MaxChildren - 1 nodes, and the tree has at most MaxDepth levels
    InplaceStack<int, 64> subtasks;
    subtasks.push( tree.rootRef() );
    while ( !subtasks.empty() )
    {
        const int r = subtasks.top();
        subtasks.pop();
        if ( r < 0 )
        {
            // boxOk is monotone, so a slot passing with the inner bound or failing with the outer one is decided without the exact box
            AABBTree::SimdBoxes inner( noInit ), outer( noInit );
            int bad;
            const int used = leafBoxBounds( tree.simdLeaves()[~r], inner, outer, bad );
            int m = boxOk( std::as_const( inner ) ) & used;
            if ( bad || m != ( boxOk( std::as_const( outer ) ) & used ) )
                m = boxOk( leafTriBoxes( tree, ~r ) ); // the unused slots have empty boxes, which do not pass
            for ( ; m; m &= m - 1 )
                if ( onTri( ~r * AABBTree::MaxLeafTris + Simd4::lowestBit( m ) ) == Processing::Stop )
                    return;
            continue;
        }
        const auto & sn = tree.simdNodes()[r];
        const int m = boxOk( static_cast<const AABBTree::SimdBoxes &>( sn ) );
        for ( int i = AABBTree::MaxChildren - 1; i >= 0; --i )
            if ( m & ( 1 << i ) )
                subtasks.push( sn.child[i] );
    }
}

} // namespace MR
