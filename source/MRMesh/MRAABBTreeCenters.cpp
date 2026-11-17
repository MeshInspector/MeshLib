#include "MRAABBTreeCenters.hpp"
#include "MRBitCast.h"
#include "MRNoDefInit.h"
#include "MRParallelFor.h"
#include "MRVector.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace MR
{

Box3f includeCenters( Box3f box, const TreeCenter * d, int num )
{
    // the lanes of a center are the bits of its index (ignored), x, y, z; two pairs of boxes for the processor to include two centers at once
    static_assert( offsetof( TreeCenter, c ) == 4 );
#if defined(__x86_64__) || defined(_M_X64)
    // min( v, box ) = v < box ? v : box skips a NaN in v as Box::include does
    __m128 lo0 = _mm_set_ps( box.min.z, box.min.y, box.min.x, 0 ), hi0 = _mm_set_ps( box.max.z, box.max.y, box.max.x, 0 );
    __m128 lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const __m128 a = _mm_loadu_ps( (const float *)( d + i ) ), b = _mm_loadu_ps( (const float *)( d + i + 1 ) );
        lo0 = _mm_min_ps( a, lo0 ); hi0 = _mm_max_ps( a, hi0 );
        lo1 = _mm_min_ps( b, lo1 ); hi1 = _mm_max_ps( b, hi1 );
    }
    if ( i < num )
    {
        const __m128 a = _mm_loadu_ps( (const float *)( d + i ) );
        lo0 = _mm_min_ps( a, lo0 ); hi0 = _mm_max_ps( a, hi0 );
    }
    alignas( 16 ) float lo[4], hi[4];
    _mm_store_ps( lo, _mm_min_ps( lo0, lo1 ) );
    _mm_store_ps( hi, _mm_max_ps( hi0, hi1 ) );
#elif defined(__aarch64__) || defined(_M_ARM64)
    // minNum and maxNum skip a quiet NaN as Box::include does (computed centers are never signaling NaNs)
    float lo[4] = { 0, box.min.x, box.min.y, box.min.z }, hi[4] = { 0, box.max.x, box.max.y, box.max.z };
    float32x4_t lo0 = vld1q_f32( lo ), hi0 = vld1q_f32( hi ), lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const float32x4_t a = vld1q_f32( (const float *)( d + i ) ), b = vld1q_f32( (const float *)( d + i + 1 ) );
        lo0 = vminnmq_f32( lo0, a ); hi0 = vmaxnmq_f32( hi0, a );
        lo1 = vminnmq_f32( lo1, b ); hi1 = vmaxnmq_f32( hi1, b );
    }
    if ( i < num )
    {
        const float32x4_t a = vld1q_f32( (const float *)( d + i ) );
        lo0 = vminnmq_f32( lo0, a ); hi0 = vmaxnmq_f32( hi0, a );
    }
    vst1q_f32( lo, vminnmq_f32( lo0, lo1 ) );
    vst1q_f32( hi, vmaxnmq_f32( hi0, hi1 ) );
#elif defined(__wasm_simd128__)
    // pmin( box, v ) = v < box ? v : box skips a NaN in v as Box::include does
    alignas( 16 ) float lo[4] = { 0, box.min.x, box.min.y, box.min.z }, hi[4] = { 0, box.max.x, box.max.y, box.max.z };
    v128_t lo0 = wasm_v128_load( lo ), hi0 = wasm_v128_load( hi ), lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const v128_t a = wasm_v128_load( (const float *)( d + i ) ), b = wasm_v128_load( (const float *)( d + i + 1 ) );
        lo0 = wasm_f32x4_pmin( lo0, a ); hi0 = wasm_f32x4_pmax( hi0, a );
        lo1 = wasm_f32x4_pmin( lo1, b ); hi1 = wasm_f32x4_pmax( hi1, b );
    }
    if ( i < num )
    {
        const v128_t a = wasm_v128_load( (const float *)( d + i ) );
        lo0 = wasm_f32x4_pmin( lo0, a ); hi0 = wasm_f32x4_pmax( hi0, a );
    }
    wasm_v128_store( lo, wasm_f32x4_pmin( lo0, lo1 ) );
    wasm_v128_store( hi, wasm_f32x4_pmax( hi0, hi1 ) );
#else
    Box3f b1;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        box.include( d[i].c );
        b1.include( d[i + 1].c );
    }
    if ( i < num )
        box.include( d[i].c );
    box.include( b1 );
    return box;
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64) || defined(__wasm_simd128__)
    return Box3f( Vector3f( lo[1], lo[2], lo[3] ), Vector3f( hi[1], hi[2], hi[3] ) );
#endif
}

namespace
{

/// the order of centers along an axis, ties by index; without branches to be fast on unordered centers
struct CenterLess
{
    int axis;
    bool operator()( const TreeCenter & a, const TreeCenter & b ) const
    {
        return bool( int( a.c[axis] < b.c[axis] ) | ( int( !( b.c[axis] < a.c[axis] ) ) & int( a.i < b.i ) ) );
    }
};

/// returns the first of the centers in [l, r) not less than p after moving all less ones before it:
/// blocks of centers are scanned for misplaced ones without branches, then the misplaced ones are swapped in pairs
TreeCenter * partition( TreeCenter * l, TreeCenter * r, const TreeCenter & p, CenterLess less )
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
    // at most two blocks are left
    for ( TreeCenter * i = l; i != r; ++i )
    {
        const bool lt = less( *i, p );
        std::swap( *i, *l );
        l += lt;
    }
    return l;
}

/// sorts up to 16 centers by counting for each one the centers before it, without branches;
/// the keys are the bits of the coordinate ordered as signed integers, then the index
void sortSmall( TreeCenter * data, int n, int axis )
{
    assert( n <= 16 );
    alignas( 16 ) std::int32_t key[16], id[16];
    for ( int j = 0; j < 16; ++j )
    {
        if ( j < n )
        {
            // + 0.0f makes -0 equal to +0
            const auto u = bit_cast<std::uint32_t>( data[j].c[axis] + 0.0f );
            key[j] = std::int32_t( u ^ ( std::uint32_t( std::int32_t( u ) >> 31 ) & 0x7fffffffu ) );
            id[j] = data[j].i;
        }
        else
        {
            // after all centers
            key[j] = 0x7fffffff;
            id[j] = 0x7fffffff;
        }
    }
    const int numVecs = n <= 8 ? 2 : 4;
    NoDefInit<TreeCenter> src[16];
    std::copy( data, data + n, src );
    for ( int j = 0; j < n; ++j )
    {
#if defined(__x86_64__) || defined(_M_X64)
        const __m128i kj = _mm_set1_epi32( key[j] ), ij = _mm_set1_epi32( id[j] );
        __m128i acc = _mm_setzero_si128();
        for ( int v = 0; v < numVecs; ++v )
        {
            const __m128i k = _mm_load_si128( (const __m128i*)key + v ), i = _mm_load_si128( (const __m128i*)id + v );
            acc = _mm_sub_epi32( acc, _mm_or_si128( _mm_cmplt_epi32( k, kj ), _mm_and_si128( _mm_cmpeq_epi32( k, kj ), _mm_cmplt_epi32( i, ij ) ) ) );
        }
        acc = _mm_add_epi32( acc, _mm_shuffle_epi32( acc, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
        acc = _mm_add_epi32( acc, _mm_shuffle_epi32( acc, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
        const int rank = _mm_cvtsi128_si32( acc );
#elif defined(__aarch64__) || defined(_M_ARM64)
        const int32x4_t kj = vdupq_n_s32( key[j] ), ij = vdupq_n_s32( id[j] );
        uint32x4_t acc = vdupq_n_u32( 0 );
        for ( int v = 0; v < numVecs; ++v )
        {
            const int32x4_t k = vld1q_s32( key + 4 * v ), i = vld1q_s32( id + 4 * v );
            acc = vsubq_u32( acc, vorrq_u32( vcltq_s32( k, kj ), vandq_u32( vceqq_s32( k, kj ), vcltq_s32( i, ij ) ) ) );
        }
        const int rank = int( vaddvq_u32( acc ) );
#elif defined(__wasm_simd128__)
        const v128_t kj = wasm_i32x4_splat( key[j] ), ij = wasm_i32x4_splat( id[j] );
        v128_t acc = wasm_i32x4_splat( 0 );
        for ( int v = 0; v < numVecs; ++v )
        {
            const v128_t k = wasm_v128_load( key + 4 * v ), i = wasm_v128_load( id + 4 * v );
            acc = wasm_i32x4_sub( acc, wasm_v128_or( wasm_i32x4_lt( k, kj ), wasm_v128_and( wasm_i32x4_eq( k, kj ), wasm_i32x4_lt( i, ij ) ) ) );
        }
        const int rank = wasm_i32x4_extract_lane( acc, 0 ) + wasm_i32x4_extract_lane( acc, 1 ) + wasm_i32x4_extract_lane( acc, 2 ) + wasm_i32x4_extract_lane( acc, 3 );
#else
        int rank = 0;
        for ( int i = 0; i < 4 * numVecs; ++i )
            rank += int( key[i] < key[j] ) | ( int( key[i] == key[j] ) & int( id[i] < id[j] ) );
#endif
        data[rank] = src[j];
    }
}

} //anonymous namespace

void selectCenter( TreeCenter * first, TreeCenter * nth, TreeCenter * last, int axis )
{
    const CenterLess less{ axis };
    constexpr int cSmall = 16;
    int budget = 2; // partitions before std::nth_element, about twice the expected number
    for ( auto n = last - first; n > cSmall; n >>= 1 )
        budget += 2;
    while ( last - first > cSmall )
    {
        if ( --budget < 0 )
        {
            // unlucky pivots
            std::nth_element( first, nth, last, less );
            return;
        }
        const auto med3 = [less]( TreeCenter * a, TreeCenter * b, TreeCenter * c )
        {
            if ( less( *b, *a ) )
                std::swap( a, b );
            return less( *c, *b ) ? ( less( *c, *a ) ? a : c ) : b;
        };
        const auto n = last - first;
        TreeCenter * const m = first + n / 2;
        TreeCenter * pivot;
        if ( n > 128 )
        {
            const auto s = n / 8;
            pivot = med3( med3( first, first + s, first + 2 * s ), med3( m - s, m, m + s ), med3( last - 1 - 2 * s, last - 1 - s, last - 1 ) );
        }
        else
            pivot = med3( first, m, last - 1 );
        std::swap( *first, *pivot );
        TreeCenter * const mid = partition( first + 1, last, *first, less ) - 1;
        std::swap( *first, *mid );
        if ( mid == nth )
            return;
        if ( nth < mid )
            last = mid;
        else
            first = mid + 1;
    }
    sortSmall( first, int( last - first ), axis );
}

namespace
{

/// splitCentersByBuckets for the center of order nth; depth counts the buckets split again
bool splitByBuckets( TreeCenter * src, TreeCenter * dst, int num, int nth, int axis, float lo, float hi, int chunkSize, int depth, Box3f * halfBoxes )
{
    constexpr int K = 256;
    if ( !( hi > lo ) )
        return false;
    const float scale = float( K ) / ( hi - lo );
    if ( !std::isfinite( scale ) )
        return false;
    // the buckets of the centers do not decrease along the axis
    const auto bucket = [lo, scale, axis]( const TreeCenter & x ) { const float t = ( x.c[axis] - lo ) * scale; return t < float( K ) ? int( t ) : K - 1; };
    const int numChunks = 1 + ( num - 1 ) / chunkSize;
    Vector<int> hists( size_t( numChunks ) * K );
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        // several counts for the processor to count several centers at once: neighbor centers are often in the same bucket
        int h[4][K] = {};
        const int b = ch * chunkSize, e = std::min( num, b + chunkSize );
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
        if ( numLess + numMed > nth )
            break;
        numLess += numMed;
    }

    // the first positions of each chunk's centers in the three parts of dst
    struct Pos { int l, m, g; };
    Vector<Pos, int> pos( numChunks );
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
        next.g += std::min( num - ch * chunkSize, chunkSize ) - l - m;
    }
    // the boxes of the centers moved to the smaller and to the bigger parts, by chunks
    Vector<Box3f, int> chunkBoxes( 2 * numChunks );
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        TreeCenter * pl = dst + pos[ch].l;
        TreeCenter * pm = dst + pos[ch].m;
        TreeCenter * pg = dst + pos[ch].g;
        Box3f bl, bg;
        // by blocks, to include the moved centers in the boxes while they are in the cache
        constexpr int cBlock = 2048;
        for ( int i = ch * chunkSize, e = std::min( num, i + chunkSize ); i < e; )
        {
            TreeCenter * const pl0 = pl, * const pg0 = pg;
            for ( const int be = std::min( e, i + cBlock ); i < be; ++i )
            {
                const int b = bucket( src[i] );
                const bool isL = b < mb, isG = b > mb;
                TreeCenter * d = isL ? pl : pm;
                d = isG ? pg : d;
                *d = src[i];
                pl += isL;
                pg += isG;
                pm += !isL & !isG;
            }
            bl = includeCenters( bl, pl0, int( pl - pl0 ) );
            bg = includeCenters( bg, pg0, int( pg - pg0 ) );
        }
        chunkBoxes[2 * ch] = bl;
        chunkBoxes[2 * ch + 1] = bg;
    } );
    halfBoxes[0] = halfBoxes[1] = Box3f();
    for ( int ch = 0; ch < numChunks; ++ch )
    {
        halfBoxes[0].include( chunkBoxes[2 * ch] );
        halfBoxes[1].include( chunkBoxes[2 * ch + 1] );
    }

    TreeCenter * const med = dst + numLess;
    const int medNth = nth - numLess;
    // a bucket with most of the centers (e.g. all but a far outlier) is split again by its own buckets, in parallel unlike selectCenter;
    // the positions of the moved centers in src are free
    if ( depth < 4 && numMed >= 1024 && numMed > num / 8 )
    {
        const int medChunks = 1 + ( numMed - 1 ) / chunkSize;
        Vector<Box3f, int> boxes( medChunks );
        ParallelFor( 0, medChunks, [&]( int ch )
        {
            const int b = ch * chunkSize;
            boxes[ch] = includeCenters( Box3f(), med + b, std::min( numMed - b, chunkSize ) );
        } );
        Box3f box;
        for ( const auto & b : boxes )
            box.include( b );
        TreeCenter * const tmp = src + numLess;
        Box3f medHalves[2];
        if ( splitByBuckets( med, tmp, numMed, medNth, axis, box.min[axis], box.max[axis], chunkSize, depth + 1, medHalves ) )
        {
            ParallelFor( 0, medChunks, [&]( int ch )
            {
                const int b = ch * chunkSize;
                std::copy( tmp + b, tmp + std::min( numMed, b + chunkSize ), med + b );
            } );
            halfBoxes[0].include( medHalves[0] );
            halfBoxes[1].include( medHalves[1] );
            return true;
        }
    }
    selectCenter( med, med + medNth, med + numMed, axis );
    halfBoxes[0] = includeCenters( halfBoxes[0], med, medNth );
    halfBoxes[1] = includeCenters( halfBoxes[1], med + medNth, numMed - medNth );
    return true;
}

} //anonymous namespace

bool splitCentersByBuckets( TreeCenter * src, TreeCenter * dst, int num, int axis, float lo, float hi, int chunkSize, Box3f * halfBoxes )
{
    return splitByBuckets( src, dst, num, num / 2, axis, lo, hi, chunkSize, 0, halfBoxes );
}

} //namespace MR
