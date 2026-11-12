#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRBitCast.h"
#include <algorithm>
#include <cmath>
#include <tbb/parallel_invoke.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace MR
{

namespace
{

/// the box of a triangle of AABBTree, micro expanded to avoid leaks due to float errors
Box3f wideTriBox( const Mesh & mesh, const AABBTree::LeafTri & lt )
{
#if defined(__x86_64__) || defined(_M_X64)
    // the lanes of a point: x, y, z, z
    const auto load = [&mesh]( VertId v )
    {
        const float * p = &mesh.points[v].x;
        return _mm_movelh_ps( _mm_castpd_ps( _mm_load_sd( (const double *)p ) ), _mm_load1_ps( p + 2 ) );
    };
    const __m128 a = load( lt.verts[0] ), b = load( lt.verts[1] ), c = load( lt.verts[2] );
    const __m128i lo = _mm_castps_si128( _mm_min_ps( _mm_min_ps( a, b ), c ) );
    const __m128i hi = _mm_castps_si128( _mm_max_ps( _mm_max_ps( a, b ), c ) );
    // as in nextDown and nextUp: if 0 < |x| < max in all lanes, then the closest values around x differ from it by one in the bits
    const __m128i one = _mm_set1_epi32( 1 );
    const __m128i absMask = _mm_set1_epi32( 0x7fffffff );
    const __m128i signBit = _mm_set1_epi32( int( 0x80000000u ) ); // to compare unsigned integers as signed ones
    const __m128i limit = _mm_set1_epi32( int( ( 0x7f7fffffu - 1 ) ^ 0x80000000u ) );
    const auto fast = [&]( __m128i x ) { return _mm_cmplt_epi32( _mm_xor_si128( _mm_sub_epi32( _mm_and_si128( x, absMask ), one ), signBit ), limit ); };
    if ( _mm_movemask_ps( _mm_castsi128_ps( _mm_and_si128( fast( lo ), fast( hi ) ) ) ) == 15 )
    {
        // minus one for positive and plus one for negative lo, plus one for positive and minus one for negative hi
        alignas( 16 ) float l[4], h[4];
        _mm_store_ps( l, _mm_castsi128_ps( _mm_sub_epi32( lo, _mm_or_si128( _mm_srai_epi32( lo, 31 ), one ) ) ) );
        _mm_store_ps( h, _mm_castsi128_ps( _mm_add_epi32( hi, _mm_or_si128( _mm_srai_epi32( hi, 31 ), one ) ) ) );
        return Box3f( Vector3f( l[0], l[1], l[2] ), Vector3f( h[0], h[1], h[2] ) );
    }
#endif
    Box3f box;
    for ( auto v : lt.verts )
        box.include( mesh.points[v] );
    return box.insignificantlyExpanded();
}

/// a triangle with its vertices and box, both found in one pass over the topology
struct BoxedTri
{
    AABBTree::LeafTri tri;
    Box3f box;

    explicit BoxedTri( NoInit ) noexcept : tri{ FaceId( noInit ), { VertId( noInit ), VertId( noInit ), VertId( noInit ) } }, box( noInit ) { }
};

/// sorted instead of BoxedTri: smaller and compared by one coordinate
struct Center
{
    int i; ///< in boxedTris
    Vector3f c; ///< doubled center of the box

    Center( int i, const Vector3f & c ) noexcept : i( i ), c( c ) { }
    explicit Center( NoInit ) noexcept : c( noInit ) { }
};
static_assert( sizeof( Center ) == 16 );

/// the triangles with their boxes, and the doubled centers of these boxes
Buffer<BoxedTri> makeBoxedTris( const MeshPart & mp, Buffer<Center> & centers )
{
    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return {};

    Buffer<BoxedTri> boxedTris( numFaces );
    centers = Buffer<Center>( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            boxedTris[n++].tri.face = f;
    }

    const auto & topology = mp.mesh.topology;
    ParallelFor( 0, numFaces, [&] ( int i )
    {
        auto & bt = boxedTris[i];
        const FaceId f = packed ? FaceId( i ) : bt.tri.face;
        // topology.getTriVerts( f ) inlined
        const EdgeId a = topology.edgeWithLeft( f );
        const EdgeId b = topology.prev( a.sym() );
        bt.tri = { f, { topology.org( a ), topology.org( b ), topology.dest( b ) } };
        bt.box = wideTriBox( mp.mesh, bt.tri );
        centers[i] = Center( i, bt.box.min + bt.box.max );
    } );
    return boxedTris;
}

/// returns given box including the centers d[0..num)
Box3f includeCenters( Box3f box, const Center * d, int num )
{
#if defined(__x86_64__) || defined(_M_X64)
    // the lanes of a center: the bits of its index (ignored), x, y, z
    __m128 lo0 = _mm_set_ps( box.min.z, box.min.y, box.min.x, 0 ), hi0 = _mm_set_ps( box.max.z, box.max.y, box.max.x, 0 );
    __m128 lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const __m128 a = _mm_loadu_ps( &d[i].c.x - 1 );
        const __m128 b = _mm_loadu_ps( &d[i + 1].c.x - 1 );
        lo0 = _mm_min_ps( lo0, a ); hi0 = _mm_max_ps( hi0, a );
        lo1 = _mm_min_ps( lo1, b ); hi1 = _mm_max_ps( hi1, b );
    }
    if ( i < num )
    {
        const __m128 a = _mm_loadu_ps( &d[i].c.x - 1 );
        lo0 = _mm_min_ps( lo0, a ); hi0 = _mm_max_ps( hi0, a );
    }
    alignas( 16 ) float lo[4], hi[4];
    _mm_store_ps( lo, _mm_min_ps( lo0, lo1 ) );
    _mm_store_ps( hi, _mm_max_ps( hi0, hi1 ) );
    return Box3f( Vector3f( lo[1], lo[2], lo[3] ), Vector3f( hi[1], hi[2], hi[3] ) );
#else
    // independent boxes for the processor to include several centers at once
    Box3f b1, b2, b3;
    int i = 0;
    for ( ; i + 4 <= num; i += 4 )
    {
        box.include( d[i].c );
        b1.include( d[i + 1].c );
        b2.include( d[i + 2].c );
        b3.include( d[i + 3].c );
    }
    for ( ; i < num; ++i )
        box.include( d[i].c );
    box.include( b1 );
    box.include( b2 );
    box.include( b3 );
    return box;
#endif
}

/// the order of centers along an axis, ties by index; without branches to be fast on unordered centers
struct CenterLess
{
    int axis;
    bool operator()( const Center & a, const Center & b ) const
    {
        return bool( int( a.c[axis] < b.c[axis] ) | ( int( !( b.c[axis] < a.c[axis] ) ) & int( a.i < b.i ) ) );
    }
};

/// returns the first of the centers in [l, r) not less than p after moving all less ones before it:
/// blocks of centers are scanned for misplaced ones without branches, then the misplaced ones are swapped in pairs
Center * partition( Center * l, Center * r, const Center & p, CenterLess less )
{
    constexpr int B = 64;
    unsigned char offL[B], offR[B];
    int nL = 0, nR = 0, sL = 0, sR = 0;
    while ( r - l > 2 * B )
    {
        if ( nL == 0 )
        {
            sL = 0;
            for ( int i = 0; i < B; ++i )
            {
                offL[nL] = (unsigned char)i;
                nL += !less( l[i], p );
            }
        }
        if ( nR == 0 )
        {
            sR = 0;
            for ( int i = 0; i < B; ++i )
            {
                offR[nR] = (unsigned char)( i + 1 );
                nR += less( *( r - ( i + 1 ) ), p );
            }
        }
        const int n = std::min( nL, nR );
        for ( int j = 0; j < n; ++j )
            std::swap( l[offL[sL + j]], *( r - offR[sR + j] ) );
        nL -= n;
        nR -= n;
        sL += n;
        sR += n;
        if ( nL == 0 )
            l += B;
        if ( nR == 0 )
            r -= B;
    }
    // at most three blocks are left
    for ( Center * i = l; i != r; ++i )
    {
        const bool lt = less( *i, p );
        std::swap( *i, *l );
        l += lt;
    }
    return l;
}

/// sorts up to 16 centers by a sorting network on their keys: the bits of the coordinate ordered as unsigned integers, then the index
void sortSmall( Center * data, int n, int axis )
{
    assert( n <= 16 );
    struct Pair { unsigned char a, b; };
    // Batcher's odd-even merge sort
    static constexpr Pair net8[] = { {0,1}, {2,3}, {4,5}, {6,7}, {0,2}, {1,3}, {4,6}, {5,7}, {1,2}, {5,6}, {0,4}, {1,5}, {2,6}, {3,7}, {2,4}, {3,5}, {1,2}, {3,4}, {5,6} };
    static constexpr Pair net16[] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {12,13}, {14,15}, {0,2}, {1,3}, {4,6}, {5,7}, {8,10}, {9,11}, {12,14}, {13,15},
        {1,2}, {5,6}, {9,10}, {13,14}, {0,4}, {1,5}, {2,6}, {3,7}, {8,12}, {9,13}, {10,14}, {11,15}, {2,4}, {3,5}, {10,12}, {11,13}, {1,2}, {3,4}, {5,6},
        {9,10}, {11,12}, {13,14}, {0,8}, {1,9}, {2,10}, {3,11}, {4,12}, {5,13}, {6,14}, {7,15}, {4,8}, {5,9}, {6,10}, {7,11}, {2,4}, {3,5}, {6,8}, {7,9},
        {10,12}, {11,13}, {1,2}, {3,4}, {5,6}, {7,8}, {9,10}, {11,12}, {13,14} };
    std::uint64_t key[16];
    unsigned char pos[16];
    for ( int j = 0; j < 16; ++j )
    {
        if ( j < n )
        {
            // + 0.0f makes -0 equal to +0
            const auto u = bit_cast<std::uint32_t>( data[j].c[axis] + 0.0f );
            const auto ordered = u ^ ( std::uint32_t( std::int32_t( u ) >> 31 ) | 0x80000000u );
            key[j] = ( std::uint64_t( ordered ) << 32 ) | std::uint32_t( data[j].i );
        }
        else
            key[j] = ~std::uint64_t( 0 ); // after all centers
        pos[j] = (unsigned char)j;
    }
    const auto sortNet = [&]( const auto & net )
    {
        for ( auto [a, b] : net )
        {
            const auto ka = key[a], kb = key[b];
            const auto pa = pos[a], pb = pos[b];
            const bool swap = kb < ka;
            key[a] = swap ? kb : ka;
            key[b] = swap ? ka : kb;
            pos[a] = swap ? pb : pa;
            pos[b] = swap ? pa : pb;
        }
    };
    if ( n <= 8 )
        sortNet( net8 );
    else
        sortNet( net16 );
    NoDefInit<Center> sorted[16];
    for ( int j = 0; j < n; ++j )
        sorted[j] = data[pos[j]];
    std::copy( sorted, sorted + n, data );
}

/// moves the center of order nth to its place, the less ones before it and the greater ones after it
void select( Center * first, Center * nth, Center * last, CenterLess less )
{
    int budget = 2 * ( 1 + int( std::log2( double( last - first ) + 1 ) ) );
    while ( last - first > 16 )
    {
        if ( --budget < 0 )
        {
            // unlucky pivots
            std::nth_element( first, nth, last, less );
            return;
        }
        const auto med3 = [less]( Center * a, Center * b, Center * c )
        {
            if ( less( *b, *a ) )
                std::swap( a, b );
            return less( *c, *b ) ? ( less( *c, *a ) ? a : c ) : b;
        };
        const auto n = last - first;
        Center * const m = first + n / 2;
        Center * pivot;
        if ( n > 128 )
        {
            const auto s = n / 8;
            pivot = med3( med3( first, first + s, first + 2 * s ), med3( m - s, m, m + s ), med3( last - 1 - 2 * s, last - 1 - s, last - 1 ) );
        }
        else
            pivot = med3( first, m, last - 1 );
        std::swap( *first, *pivot );
        Center * const mid = partition( first + 1, last, *first, less ) - 1;
        std::swap( *first, *mid );
        if ( mid == nth )
            return;
        if ( nth < mid )
            last = mid;
        else
            first = mid + 1;
    }
    sortSmall( first, int( last - first ), less.axis );
}

/// splits each AABBTree node in two binary levels, each at the median of the triangle box centers
/// along the largest extent of these centers; so the nodes at binary depth d have ( numTris >> d ) or ( numTris >> d ) + 1 triangles
class AABBTreeWideMaker
{
public:
    using Node = AABBTree::Node;

    /// fills nodes, and the triangles and their boxes in leaf order
    AABBTreeWideMaker( const Buffer<BoxedTri> & boxedTris, Buffer<Center> & centers,
        Vector<Node, NodeId> & nodes, Vector<AABBTree::LeafTri> & leafTris, Vector<Box3f> & leafBoxes )
        : boxedTris_( boxedTris ), centers_( centers ), nodes_( nodes ), leafTris_( leafTris ), leafBoxes_( leafBoxes )
    {
        numTris_ = (int)boxedTris_.size();
        for ( auto & c : cnt_ )
            c = { -1, -1 };
        nodes_.resize( countNodes_( 0, numTris_ ) );
        parallelSplits_ = numTris_ >= cParallelSplitNum
            && tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) > 1
            && tbb::this_task_arena::max_concurrency() > 1;
        if ( numTris_ >= cHistSplitNum )
            scratch_ = Buffer<Center>( numTris_ );
        make_( NodeId( 0 ), NodeId( 1 ), { 0, { 0, numTris_, false } } );
    }

private:
    /// centers [first, first + num) either in centers_ or at the same positions in scratch_
    struct Range
    {
        int first, num;
        bool inScratch;
    };
    struct Child
    {
        int depth;
        Range r;
    };

    Center * data_( const Range & r ) const { return ( r.inScratch ? scratch_.data() : centers_.data() ) + r.first; }

    // the number of nodes in the subtree of a node at binary depth d with num triangles
    int countNodes_( int d, int num )
    {
        int & res = cnt_[d][num - ( numTris_ >> d )];
        if ( res >= 0 )
            return res;
        if ( num <= AABBTree::MaxLeafTris )
            return res = 1;
        res = 1;
        const int halves[2] = { num / 2, num - num / 2 };
        for ( int h : halves )
        {
            if ( h <= AABBTree::MaxLeafTris )
                res += countNodes_( d + 1, h );
            else
                res += countNodes_( d + 2, h / 2 ) + countNodes_( d + 2, h - h / 2 );
        }
        return res;
    }
    int numNodes_( int d, int num ) const { return cnt_[d][num - ( numTris_ >> d )]; }

    Box3f centersBox_( const Range & r ) const
    {
        const Center * d = data_( r );
        if ( !parallelSplits_ || r.num < cParallelSplitNum )
            return includeCenters( Box3f(), d, r.num );
        return tbb::parallel_reduce( tbb::blocked_range<int>( 0, r.num, 1 << 14 ), Box3f(),
            [d]( const tbb::blocked_range<int> & br, Box3f b ) { return includeCenters( b, d + br.begin(), (int)br.size() ); },
            []( Box3f a, const Box3f & b ) { a.include( b ); return a; } );
    }

    // [first, first + num / 2) get the smaller centers along the largest extent of the centers; ties are ordered by index
    void split_( Range & r )
    {
        const auto box = centersBox_( r );
        const auto sz = box.size();
        const int axis = sz.x >= sz.y ? ( sz.x >= sz.z ? 0 : 2 ) : ( sz.y >= sz.z ? 1 : 2 );
        const CenterLess less{ axis };
        if ( r.num >= cHistSplitNum && histSplit_( r, less, box.min[axis], box.max[axis] ) )
            return;
        Center * const data = data_( r );
        select( data, data + r.num / 2, data + r.num, less );
    }

    // splits by the buckets of the centers between lo and hi: the counts of the centers in the buckets find the bucket of the median,
    // one pass moves the centers of the smaller buckets, of the median's bucket and of the bigger buckets to their places in the other buffer,
    // and only the median's bucket is left to select in; returns false if the buckets cannot be made
    bool histSplit_( Range & r, CenterLess less, float lo, float hi )
    {
        constexpr int K = 256;
        const float scale = float( K ) / ( hi - lo );
        if ( !( hi > lo ) || !std::isfinite( scale ) )
            return false;
        const int axis = less.axis;
        // the buckets of the centers do not decrease along the axis
        const auto bucket = [lo, scale, axis]( const Center & x ) { const float t = ( x.c[axis] - lo ) * scale; return t < float( K ) ? int( t ) : K - 1; };
        const Center * const src = data_( r );
        Center * const dst = data_( { r.first, r.num, !r.inScratch } );
        const int mid = r.num / 2;

        // one chunk if not parallel
        const int chunk = parallelSplits_ && r.num >= cParallelSplitNum ? cChunk : r.num;
        const int numChunks = ( r.num + chunk - 1 ) / chunk;
        std::vector<int> hists( size_t( numChunks ) * K );
        ParallelFor( 0, numChunks, [&]( int ch )
        {
            // several counts for the processor to count several centers at once: neighbor centers are often in the same bucket
            int h[4][K] = {};
            const int b = ch * chunk, e = std::min( r.num, b + chunk );
            int i = b;
            for ( ; i + 4 <= e; i += 4 )
            {
                ++h[0][bucket( src[i] )];
                ++h[1][bucket( src[i + 1] )];
                ++h[2][bucket( src[i + 2] )];
                ++h[3][bucket( src[i + 3] )];
            }
            for ( ; i < e; ++i )
                ++h[0][bucket( src[i] )];
            int * const res = hists.data() + size_t( ch ) * K;
            for ( int k = 0; k < K; ++k )
                res[k] = h[0][k] + h[1][k] + h[2][k] + h[3][k];
        } );

        int mb = 0, numLess = 0, numMed = 0;
        for ( ;; ++mb )
        {
            numMed = 0;
            for ( int ch = 0; ch < numChunks; ++ch )
                numMed += hists[size_t( ch ) * K + mb];
            if ( numLess + numMed > mid )
                break;
            numLess += numMed;
        }

        // the first positions of each chunk's centers in the three parts of dst
        struct Pos { int l, m, g; };
        std::vector<Pos> pos( numChunks );
        Pos next{ 0, numLess, numLess + numMed };
        for ( int ch = 0; ch < numChunks; ++ch )
        {
            pos[ch] = next;
            const int * h = hists.data() + size_t( ch ) * K;
            int l = 0;
            for ( int k = 0; k < mb; ++k )
                l += h[k];
            const int m = h[mb];
            next.l += l;
            next.m += m;
            next.g += std::min( r.num - ch * chunk, chunk ) - l - m;
        }
        ParallelFor( 0, numChunks, [&]( int ch )
        {
            Center * pl = dst + pos[ch].l;
            Center * pm = dst + pos[ch].m;
            Center * pg = dst + pos[ch].g;
            for ( int i = ch * chunk, e = std::min( r.num, i + chunk ); i < e; ++i )
            {
                const int b = bucket( src[i] );
                const bool isL = b < mb, isG = b > mb;
                Center * d = isL ? pl : pm;
                d = isG ? pg : d;
                *d = src[i];
                pl += isL;
                pg += isG;
                pm += !isL & !isG;
            }
        } );
        r.inScratch = !r.inScratch;
        select( dst + numLess, dst + mid, dst + numLess + numMed, less );
        return true;
    }

    // fills node n with its subtree, whose nodes below the children of n start from desc
    void make_( NodeId n, NodeId desc, Child c )
    {
        auto & node = nodes_[n];
        const int first = c.r.first, num = c.r.num;
        if ( num <= AABBTree::MaxLeafTris )
        {
            // the triangles of a leaf in the order of their indices
            Center * const leaf = data_( c.r );
            const auto cs = [leaf]( int x, int y ) { if ( leaf[y].i < leaf[x].i ) std::swap( leaf[x], leaf[y] ); };
            switch ( num )
            {
            case 2: cs( 0, 1 ); break;
            case 3: cs( 0, 1 ); cs( 1, 2 ); cs( 0, 1 ); break;
            case 4: cs( 0, 1 ); cs( 2, 3 ); cs( 0, 2 ); cs( 1, 3 ); cs( 1, 2 ); break;
            default: break;
            }
            Box3f box;
            for ( int i = 0; i < num; ++i )
            {
                const auto & bt = boxedTris_[leaf[i].i];
                box.include( bt.box );
                leafTris_[first + i] = bt.tri;
                leafBoxes_[first + i] = bt.box;
            }
            node.box = box;
            node.first = first;
            node.count = -num;
            return;
        }

        split_( c.r );
        Child halves[2] = { { c.depth + 1, { first, num / 2, c.r.inScratch } }, { c.depth + 1, { first + num / 2, num - num / 2, c.r.inScratch } } };
        const bool parallel = num >= 4096;
        if ( parallel )
            tbb::parallel_invoke( [&] { split_( halves[0].r ); }, [&] { split_( halves[1].r ); } );
        Child ch[AABBTree::MaxChildren];
        int k = 0;
        for ( auto & h : halves )
        {
            if ( h.r.num <= AABBTree::MaxLeafTris )
            {
                ch[k++] = h;
                continue;
            }
            if ( !parallel )
                split_( h.r );
            ch[k++] = { c.depth + 2, { h.r.first, h.r.num / 2, h.r.inScratch } };
            ch[k++] = { c.depth + 2, { h.r.first + h.r.num / 2, h.r.num - h.r.num / 2, h.r.inScratch } };
        }
        node.first = int( desc );
        node.count = k;

        NodeId childDesc[AABBTree::MaxChildren];
        NodeId next = desc + k;
        for ( int j = 0; j < k; ++j )
        {
            childDesc[j] = next;
            next = next + ( numNodes_( ch[j].depth, ch[j].r.num ) - 1 );
        }
        if ( !parallel )
        {
            for ( int j = 0; j < k; ++j )
                make_( desc + j, childDesc[j], ch[j] );
        }
        else
        {
            tbb::task_group group;
            for ( int j = 1; j < k; ++j )
                group.run( [&, j] { make_( desc + j, childDesc[j], ch[j] ); } );
            make_( desc, childDesc[0], ch[0] );
            group.wait();
        }

        Box3f box;
        for ( int j = 0; j < k; ++j )
            box.include( nodes_[desc + j].box );
        node.box = box;
    }

    const Buffer<BoxedTri> & boxedTris_;
    Buffer<Center> & centers_;
    Buffer<Center> scratch_; // the centers of a range move between centers_ and scratch_ by each split of the range with buckets
    static constexpr int cHistSplitNum = 1024; // smaller ranges are split by select
    static constexpr int cParallelSplitNum = 1 << 17;
    static constexpr int cChunk = 1 << 14; // the centers counted and moved by one thread in a parallel split
    bool parallelSplits_ = false; // on one thread the parallel split only adds passes over the range
    Vector<Node, NodeId> & nodes_;
    Vector<AABBTree::LeafTri> & leafTris_;
    Vector<Box3f> & leafBoxes_;
    int numTris_ = 0;
    std::array<int, 2> cnt_[64]; // by binary depth and the number of triangles minus ( numTris_ >> depth )
};

} // anonymous namespace

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER;
    Buffer<Center> centers;
    auto boxedTris = makeBoxedTris( mp, centers );
    if ( boxedTris.size() == 0 )
        return;
    leafTris_.resize( boxedTris.size() );
    leafBoxes_.resizeNoInit( boxedTris.size() );
    AABBTreeWideMaker maker( boxedTris, centers, nodes_, leafTris_, leafBoxes_ );
}

std::vector<NodeId> AABBTree::getSubtrees( int minNum ) const
{
    MR_TIMER;
    assert( minNum > 0 );
    std::vector<NodeId> res;
    if ( nodes_.empty() )
        return res;
    res.push_back( NodeId( 0 ) );
    std::vector<NodeId> tmp;
    while ( res.size() < minNum )
    {
        bool split = false;
        for ( NodeId n : res )
        {
            const auto & node = nodes_[n];
            if ( node.leaf() )
            {
                tmp.push_back( n );
                continue;
            }
            for ( int i = 0; i < node.count; ++i )
                tmp.push_back( NodeId( node.first + i ) );
            split = true;
        }
        res.swap( tmp );
        tmp.clear();
        if ( !split )
            break;
    }
    return res;
}

FaceBitSet AABBTree::getSubtreeFaces( NodeId subtreeRoot ) const
{
    MR_TIMER;
    FaceBitSet res;
    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( subtreeRoot );
    while ( !subtasks.empty() )
    {
        const auto & node = nodes_[subtasks.top()];
        subtasks.pop();
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                res.autoResizeSet( leafTris_[i].face );
            continue;
        }
        for ( int i = node.count - 1; i >= 0; --i )
            subtasks.push( NodeId( node.first + i ) );
    }
    return res;
}

NodeBitSet AABBTree::getNodesFromFaces( const FaceBitSet & faces ) const
{
    MR_TIMER;
    NodeBitSet res( nodes_.size() );
    if ( nodes_.empty() )
        return res;
    BitSetParallelForAll( res, [&]( NodeId n )
    {
        const auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            if ( faces.test( leafTris_[i].face ) )
            {
                res.set( n );
                return;
            }
        }
    } );
    // children follow their parent
    for ( auto n = nodes_.backId(); n.valid(); --n )
    {
        const auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        for ( int i = 0; i < node.count; ++i )
        {
            if ( res.test( NodeId( node.first + i ) ) )
            {
                res.set( n );
                break;
            }
        }
    }
    return res;
}

void AABBTree::getLeafOrder( FaceBMap & faceMap ) const
{
    MR_TIMER;
    for ( size_t i = 0; i < leafTris_.size(); ++i )
        faceMap.b[leafTris_[i].face] = FaceId( i );
    faceMap.tsize = decltype( faceMap.tsize )( leafTris_.size() );
}

void AABBTree::renumber( const FaceBMap & faceMap, const VertBMap & vertMap )
{
    MR_TIMER;
    ParallelFor( leafTris_, [&]( size_t i )
    {
        auto & lt = leafTris_[i];
        lt.face = faceMap.b[lt.face];
        for ( auto & v : lt.verts )
            v = vertMap.b[v];
    } );
}

void AABBTree::flipOrientation( const FaceBitSet * region )
{
    MR_TIMER;
    ParallelFor( leafTris_, [&]( size_t i )
    {
        auto & lt = leafTris_[i];
        if ( !region || region->test( lt.face ) )
            std::swap( lt.verts[0], lt.verts[1] );
    } );
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet * changedVerts )
{
    MR_TIMER;
    if ( nodes_.empty() )
        return;
    FaceBitSet changedFaces;
    if ( changedVerts )
        changedFaces = getIncidentFaces( mesh.topology, *changedVerts );

    // the boxes of the leaves with changed triangles
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId n )
    {
        auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        bool changed = false;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            if ( changedVerts && !changedFaces.test( leafTris_[i].face ) )
                continue;
            leafBoxes_[i] = wideTriBox( mesh, leafTris_[i] );
            changed = true;
        }
        if ( !changed )
            return;
        changedNodes.set( n );
        Box3f box;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
            box.include( leafBoxes_[i] );
        node.box = box;
    } );

    // the other nodes from the last: children follow their parent
    for ( auto n = nodes_.backId(); n.valid(); --n )
    {
        auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        bool changed = false;
        for ( int i = 0; i < node.count && !changed; ++i )
            changed = changedNodes.test( NodeId( node.first + i ) );
        if ( !changed )
            continue;
        changedNodes.set( n );
        Box3f box;
        for ( int i = 0; i < node.count; ++i )
            box.include( nodes_[NodeId( node.first + i )].box );
        node.box = box;
    }
}

void AABBTree::addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old )
{
    MR_TIMER;
    if ( new2Old.empty() )
        return;
    assert( !nodes_.empty() );

    FaceBitSet splitFaces( mesh.topology.faceSize() );
    for ( const auto & [newFace, oldFace] : new2Old )
        splitFaces.set( oldFace );

    // the leaf of every split face, and the new vertices and boxes of the split faces
    std::vector<FaceId> oldFaces; // in increasing order
    oldFaces.reserve( splitFaces.count() );
    for ( auto f : splitFaces )
        oldFaces.push_back( f );
    std::vector<NodeId> leafOfOld( oldFaces.size() );
    ParallelFor( nodes_, [&]( NodeId n )
    {
        const auto & node = nodes_[n];
        if ( !node.leaf() )
            return;
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            auto & lt = leafTris_[i];
            if ( !splitFaces.test( lt.face ) )
                continue;
            lt.verts = mesh.topology.getTriVerts( lt.face );
            leafBoxes_[i] = wideTriBox( mesh, lt );
            const auto it = std::lower_bound( oldFaces.begin(), oldFaces.end(), lt.face );
            assert( it != oldFaces.end() && *it == lt.face );
            leafOfOld[it - oldFaces.begin()] = n;
        }
    } );

    // the new faces grouped by the leaf of their old face, in a deterministic order
    std::vector<std::pair<NodeId, FaceId>> leafNew;
    leafNew.reserve( new2Old.size() );
    for ( const auto & [newFace, oldFace] : new2Old )
    {
        const auto leaf = leafOfOld[std::lower_bound( oldFaces.begin(), oldFaces.end(), oldFace ) - oldFaces.begin()];
        if ( !leaf )
        {
            assert( false ); // the split face is not in this tree
            continue;
        }
        leafNew.emplace_back( leaf, newFace );
    }
    std::sort( leafNew.begin(), leafNew.end(), []( const auto & x, const auto & y ) { return std::tie( x.first, x.second ) < std::tie( y.first, y.second ); } );

    // every changed leaf becomes a node with the old leaf as the first child and the leaves of the new faces;
    // the new nodes are appended, so children have larger ids than their parents as in a constructed tree
    const auto firstNewNode = nodes_.endId();
    std::vector<NodeId> changedLeaves;
    // appends a block of up to MaxChildren nodes for the given children, every child is a leaf or a list of more than MaxChildren leaves
    auto makeChildren = [&]( NodeId parent, std::vector<Node> leaves, auto && self ) -> void
    {
        const int n = (int)leaves.size();
        const int numCh = std::min( n, MaxChildren );
        const NodeId first = nodes_.endId();
        nodes_[parent].first = int( first );
        nodes_[parent].count = numCh;
        nodes_.resize( nodes_.size() + numCh );
        if ( n <= MaxChildren )
        {
            for ( int i = 0; i < n; ++i )
                nodes_[first + i] = leaves[i];
            return;
        }
        for ( int c = 0; c < numCh; ++c )
        {
            std::vector<Node> part( leaves.begin() + c * n / numCh, leaves.begin() + ( c + 1 ) * n / numCh );
            if ( part.size() == 1 )
                nodes_[first + c] = part.front();
            else
                self( first + c, std::move( part ), self );
        }
    };
    for ( size_t k = 0; k < leafNew.size(); )
    {
        const auto leaf = leafNew[k].first;
        std::vector<Node> leaves{ nodes_[leaf] }; // the old leaf keeps its triangles
        for ( ; k < leafNew.size() && leafNew[k].first == leaf; )
        {
            Node nl;
            nl.first = (int)leafTris_.size();
            int num = 0;
            for ( ; num < MaxLeafTris && k < leafNew.size() && leafNew[k].first == leaf; ++num, ++k )
            {
                const auto f = leafNew[k].second;
                leafTris_.push_back( { f, mesh.topology.getTriVerts( f ) } );
                leafBoxes_.push_back( wideTriBox( mesh, leafTris_.back() ) );
            }
            nl.count = -num;
            leaves.push_back( nl );
        }
        makeChildren( leaf, std::move( leaves ), makeChildren );
        changedLeaves.push_back( leaf );
    }

    // the boxes of the new nodes from the last, and then of the changed leaves (now parents)
    auto updateBox = [&]( NodeId n )
    {
        auto & node = nodes_[n];
        Box3f box;
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                box.include( leafBoxes_[i] );
        }
        else
        {
            for ( int i = 0; i < node.count; ++i )
                box.include( nodes_[NodeId( node.first + i )].box );
        }
        node.box = box;
    };
    for ( auto n = nodes_.backId(); n >= firstNewNode; --n )
        updateBox( n );
    bool grown = false; // whether the subtree of some split face is not inside the box of its old leaf
    for ( auto n : changedLeaves )
    {
        const auto oldBox = nodes_[n].box;
        updateBox( n );
        grown = grown || !oldBox.contains( nodes_[n].box );
    }

    // only a new vertex outside the box of its split face (e.g. due to rounding) makes the boxes of the ancestors to grow,
    // then update all not-leaf nodes from the last to the root as in refit()
    if ( grown )
        for ( auto n = nodes_.backId(); n.valid(); --n )
            if ( !nodes_[n].leaf() )
                updateBox( n );

    // every split leaf went one level deeper: the stacks of the traversals hold the nodes of MaxDepth levels only
    std::vector<int> depth( nodes_.size(), 0 );
    int maxDepth = 0;
    for ( NodeId n( 0 ); n < nodes_.size(); ++n )
    {
        const auto & node = nodes_[n];
        if ( node.leaf() )
            continue;
        for ( int i = 0; i < node.count; ++i )
            maxDepth = std::max( maxDepth, depth[node.first + i] = depth[n] + 1 );
    }
    if ( maxDepth > MaxDepth )
    {
        FaceBitSet faces( mesh.topology.faceSize() );
        for ( const auto & lt : leafTris_ )
            faces.set( lt.face );
        *this = AABBTree( MeshPart{ mesh, &faces } );
    }
}

} //namespace MR
