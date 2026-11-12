#include "MRAABBTreeCenters.hpp"
#include "MRBitCast.h"
#include "MRNoDefInit.h"
#include "MRParallelFor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

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
#if defined(__x86_64__) || defined(_M_X64)
    // min( v, box ) = v < box ? v : box skips a NaN in v as Box::include does
    __m128 lo0 = _mm_set_ps( box.min.z, box.min.y, box.min.x, 0 ), hi0 = _mm_set_ps( box.max.z, box.max.y, box.max.x, 0 );
    __m128 lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const __m128 a = _mm_loadu_ps( &d[i].c.x - 1 ), b = _mm_loadu_ps( &d[i + 1].c.x - 1 );
        lo0 = _mm_min_ps( a, lo0 ); hi0 = _mm_max_ps( a, hi0 );
        lo1 = _mm_min_ps( b, lo1 ); hi1 = _mm_max_ps( b, hi1 );
    }
    if ( i < num )
    {
        const __m128 a = _mm_loadu_ps( &d[i].c.x - 1 );
        lo0 = _mm_min_ps( a, lo0 ); hi0 = _mm_max_ps( a, hi0 );
    }
    alignas( 16 ) float lo[4], hi[4];
    _mm_store_ps( lo, _mm_min_ps( lo0, lo1 ) );
    _mm_store_ps( hi, _mm_max_ps( hi0, hi1 ) );
#elif defined(__aarch64__) || defined(_M_ARM64)
    // minNum and maxNum skip a NaN as Box::include does
    float lo[4] = { 0, box.min.x, box.min.y, box.min.z }, hi[4] = { 0, box.max.x, box.max.y, box.max.z };
    float32x4_t lo0 = vld1q_f32( lo ), hi0 = vld1q_f32( hi ), lo1 = lo0, hi1 = hi0;
    int i = 0;
    for ( ; i + 2 <= num; i += 2 )
    {
        const float32x4_t a = vld1q_f32( &d[i].c.x - 1 ), b = vld1q_f32( &d[i + 1].c.x - 1 );
        lo0 = vminnmq_f32( lo0, a ); hi0 = vmaxnmq_f32( hi0, a );
        lo1 = vminnmq_f32( lo1, b ); hi1 = vmaxnmq_f32( hi1, b );
    }
    if ( i < num )
    {
        const float32x4_t a = vld1q_f32( &d[i].c.x - 1 );
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
        const v128_t a = wasm_v128_load( &d[i].c.x - 1 ), b = wasm_v128_load( &d[i + 1].c.x - 1 );
        lo0 = wasm_f32x4_pmin( lo0, a ); hi0 = wasm_f32x4_pmax( hi0, a );
        lo1 = wasm_f32x4_pmin( lo1, b ); hi1 = wasm_f32x4_pmax( hi1, b );
    }
    if ( i < num )
    {
        const v128_t a = wasm_v128_load( &d[i].c.x - 1 );
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
    // at most three blocks are left
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

bool splitCentersByBuckets( const TreeCenter * src, TreeCenter * dst, int num, int axis, float lo, float hi, int chunkSize )
{
    constexpr int K = 256;
    const float scale = float( K ) / ( hi - lo );
    if ( !( hi > lo ) || !std::isfinite( scale ) )
        return false;
    // the buckets of the centers do not decrease along the axis
    const auto bucket = [lo, scale, axis]( const TreeCenter & x ) { const float t = ( x.c[axis] - lo ) * scale; return t < float( K ) ? int( t ) : K - 1; };
    const int mid = num / 2;
    const int numChunks = ( num + chunkSize - 1 ) / chunkSize;
    std::vector<int> hists( size_t( numChunks ) * K );
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
        next.g += std::min( num - ch * chunkSize, chunkSize ) - l - m;
    }
    ParallelFor( 0, numChunks, [&]( int ch )
    {
        TreeCenter * pl = dst + pos[ch].l;
        TreeCenter * pm = dst + pos[ch].m;
        TreeCenter * pg = dst + pos[ch].g;
        for ( int i = ch * chunkSize, e = std::min( num, i + chunkSize ); i < e; ++i )
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
    } );
    selectCenter( dst + numLess, dst + mid, dst + numLess + numMed, axis );
    return true;
}

} //namespace MR
