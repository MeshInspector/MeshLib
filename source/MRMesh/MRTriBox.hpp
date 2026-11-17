#pragma once

#include "MRBox.h"
#include "MRAABBTreeCenters.hpp"
#include <cstddef>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace MR
{

namespace detail
{

// the insignificantly expanded box of a, b and c in the lanes x, y, z, z of lo and hi by SIMD instructions; false for the cases left to the scalar code:
// if |x| < max in all coordinates, then nextDown and nextUp of nonzero x differ from it by one in the bits,
// and of +-0 are -denorm_min and +denorm_min, which the steps below give for -0 in lo and +0 in hi
#if defined(__x86_64__) || defined(_M_X64)
#define MR_TRI_BOX_LANES
using TriBoxLanes = __m128;
inline bool expandedTriBoxLanes( const Vector3f & a, const Vector3f & b, const Vector3f & c, TriBoxLanes & outLo, TriBoxLanes & outHi )
{
    // the lanes of a point: x, y, z, z
    const auto load = []( const Vector3f & p ) { return _mm_movelh_ps( _mm_castsi128_ps( _mm_loadl_epi64( (const __m128i *)&p.x ) ), _mm_load1_ps( &p.z ) ); };
    const __m128 va = load( a ), vb = load( b ), vc = load( c );
    // a NaN in b or c is skipped as by Box::include, and a NaN in a leads to the scalar code
    const __m128i lo = _mm_castps_si128( _mm_min_ps( vc, _mm_min_ps( vb, va ) ) );
    const __m128i hi = _mm_castps_si128( _mm_max_ps( vc, _mm_max_ps( vb, va ) ) );
    const __m128i one = _mm_set1_epi32( 1 );
    const __m128i absMask = _mm_set1_epi32( 0x7fffffff );
    const __m128i signBit = _mm_set1_epi32( int( 0x80000000u ) ); // to compare unsigned integers as signed ones
    const __m128i limit = _mm_set1_epi32( int( ( 0x7f7fffffu - 1 ) ^ 0x80000000u ) );
    const auto fast = [&]( __m128i x ) { return _mm_cmplt_epi32( _mm_xor_si128( _mm_sub_epi32( _mm_and_si128( x, absMask ), one ), signBit ), limit ); };
    const auto zero = [&]( __m128i x ) { return _mm_cmpeq_epi32( _mm_and_si128( x, absMask ), _mm_setzero_si128() ); };
    const __m128i zLo = zero( lo ), zHi = zero( hi );
    if ( _mm_movemask_ps( _mm_castsi128_ps( _mm_and_si128( _mm_or_si128( fast( lo ), zLo ), _mm_or_si128( fast( hi ), zHi ) ) ) ) != 15 )
        return false;
    const __m128i lo0 = _mm_or_si128( lo, _mm_and_si128( zLo, signBit ) );
    const __m128i hi0 = _mm_andnot_si128( _mm_and_si128( zHi, signBit ), hi );
    // minus one for positive and plus one for negative lo, plus one for positive and minus one for negative hi
    outLo = _mm_castsi128_ps( _mm_sub_epi32( lo0, _mm_or_si128( _mm_srai_epi32( lo0, 31 ), one ) ) );
    outHi = _mm_castsi128_ps( _mm_add_epi32( hi0, _mm_or_si128( _mm_srai_epi32( hi0, 31 ), one ) ) );
    return true;
}
inline void storeLanes( float * p, TriBoxLanes v ) { _mm_storeu_ps( p, v ); }
#elif defined(__aarch64__) || defined(_M_ARM64)
#define MR_TRI_BOX_LANES
using TriBoxLanes = float32x4_t;
inline bool expandedTriBoxLanes( const Vector3f & a, const Vector3f & b, const Vector3f & c, TriBoxLanes & outLo, TriBoxLanes & outHi )
{
    const auto load = []( const Vector3f & p ) { return vcombine_f32( vld1_f32( &p.x ), vld1_dup_f32( &p.z ) ); };
    const float32x4_t va = load( a ), vb = load( b ), vc = load( c );
    // any NaN, quiet or signaling, in a, b or c leads to the scalar code, tested on the points themselves
    const uint32x4_t notNan = vandq_u32( vandq_u32( vceqq_f32( va, va ), vceqq_f32( vb, vb ) ), vceqq_f32( vc, vc ) );
    const uint32x4_t lo = vreinterpretq_u32_f32( vminq_f32( vminq_f32( va, vb ), vc ) );
    const uint32x4_t hi = vreinterpretq_u32_f32( vmaxq_f32( vmaxq_f32( va, vb ), vc ) );
    const uint32x4_t one = vdupq_n_u32( 1 );
    const uint32x4_t absMask = vdupq_n_u32( 0x7fffffff );
    const uint32x4_t limit = vdupq_n_u32( 0x7f7fffffu - 1 );
    const auto fast = [&]( uint32x4_t x ) { return vcltq_u32( vsubq_u32( vandq_u32( x, absMask ), one ), limit ); };
    const auto zero = [&]( uint32x4_t x ) { return vceqq_u32( vandq_u32( x, absMask ), vdupq_n_u32( 0 ) ); };
    const uint32x4_t zLo = zero( lo ), zHi = zero( hi );
    if ( vminvq_u32( vandq_u32( notNan, vandq_u32( vorrq_u32( fast( lo ), zLo ), vorrq_u32( fast( hi ), zHi ) ) ) ) == 0 )
        return false;
    const uint32x4_t signBit = vdupq_n_u32( 0x80000000u );
    const uint32x4_t lo0 = vorrq_u32( lo, vandq_u32( zLo, signBit ) );
    const uint32x4_t hi0 = vbicq_u32( hi, vandq_u32( zHi, signBit ) );
    const auto step = [&]( uint32x4_t x ) { return vorrq_u32( vreinterpretq_u32_s32( vshrq_n_s32( vreinterpretq_s32_u32( x ), 31 ) ), one ); };
    outLo = vreinterpretq_f32_u32( vsubq_u32( lo0, step( lo0 ) ) );
    outHi = vreinterpretq_f32_u32( vaddq_u32( hi0, step( hi0 ) ) );
    return true;
}
inline void storeLanes( float * p, TriBoxLanes v ) { vst1q_f32( p, v ); }
#elif defined(__wasm_simd128__)
#define MR_TRI_BOX_LANES
using TriBoxLanes = v128_t;
inline bool expandedTriBoxLanes( const Vector3f & a, const Vector3f & b, const Vector3f & c, TriBoxLanes & outLo, TriBoxLanes & outHi )
{
    const auto load = []( const Vector3f & p ) { return wasm_i32x4_shuffle( wasm_v128_load64_zero( &p.x ), wasm_v128_load32_splat( &p.z ), 0, 1, 4, 5 ); };
    const v128_t va = load( a ), vb = load( b ), vc = load( c );
    // pmin( x, v ) = v < x ? v : x skips a NaN in v as Box::include does, and a NaN in a leads to the scalar code
    const v128_t lo = wasm_f32x4_pmin( wasm_f32x4_pmin( va, vb ), vc );
    const v128_t hi = wasm_f32x4_pmax( wasm_f32x4_pmax( va, vb ), vc );
    const v128_t one = wasm_i32x4_splat( 1 );
    const v128_t absMask = wasm_i32x4_splat( 0x7fffffff );
    const v128_t limit = wasm_i32x4_splat( int( 0x7f7fffffu - 1 ) );
    const auto fast = [&]( v128_t x ) { return wasm_u32x4_lt( wasm_i32x4_sub( wasm_v128_and( x, absMask ), one ), limit ); };
    const auto zero = [&]( v128_t x ) { return wasm_i32x4_eq( wasm_v128_and( x, absMask ), wasm_i32x4_splat( 0 ) ); };
    const v128_t zLo = zero( lo ), zHi = zero( hi );
    if ( !wasm_i32x4_all_true( wasm_v128_and( wasm_v128_or( fast( lo ), zLo ), wasm_v128_or( fast( hi ), zHi ) ) ) )
        return false;
    const v128_t signBit = wasm_i32x4_splat( int( 0x80000000u ) );
    const v128_t lo0 = wasm_v128_or( lo, wasm_v128_and( zLo, signBit ) );
    const v128_t hi0 = wasm_v128_andnot( hi, wasm_v128_and( zHi, signBit ) );
    const auto step = [&]( v128_t x ) { return wasm_v128_or( wasm_i32x4_shr( x, 31 ), one ); };
    outLo = wasm_i32x4_sub( lo0, step( lo0 ) );
    outHi = wasm_i32x4_add( hi0, step( hi0 ) );
    return true;
}
inline void storeLanes( float * p, TriBoxLanes v ) { wasm_v128_store( p, v ); }
#endif

} //namespace detail

/// returns the same box as the empty box including a, b and c, insignificantly expanded, by SIMD instructions where available
inline Box3f insignificantlyExpandedTriBox( const Vector3f & a, const Vector3f & b, const Vector3f & c )
{
#ifdef MR_TRI_BOX_LANES
    detail::TriBoxLanes lo, hi;
    if ( detail::expandedTriBoxLanes( a, b, c, lo, hi ) )
    {
        float l[4], h[4];
        detail::storeLanes( l, lo );
        detail::storeLanes( h, hi );
        return Box3f( Vector3f( l[0], l[1], l[2] ), Vector3f( h[0], h[1], h[2] ) );
    }
#endif
    Box3f box;
    box.include( a );
    box.include( b );
    box.include( c );
    return box.insignificantlyExpanded();
}

/// the same box as insignificantlyExpandedTriBox and the center of triangle i (the sum of its box's corners), stored directly from the SIMD lanes
inline void insignificantlyExpandedTriBoxAndCenter( const Vector3f & a, const Vector3f & b, const Vector3f & c, int i, Box3f & box, TreeCenter & center )
{
    static_assert( sizeof( Box3f ) == 24 && offsetof( TreeCenter, c ) == 4 );
#ifdef MR_TRI_BOX_LANES
    detail::TriBoxLanes lo, hi;
    if ( detail::expandedTriBoxLanes( a, b, c, lo, hi ) )
    {
        // the box from its min and the x of its max in one store, then the y and the z of its max; the center as index, x, y, z
#if defined(__x86_64__) || defined(_M_X64)
        const __m128 t = _mm_shuffle_ps( lo, hi, _MM_SHUFFLE( 0, 0, 2, 2 ) ); // lo.z, lo.z, hi.x, hi.x
        _mm_storeu_ps( &box.min.x, _mm_shuffle_ps( lo, t, _MM_SHUFFLE( 2, 0, 1, 0 ) ) );
        _mm_storel_pi( (__m64 *)&box.max.y, _mm_shuffle_ps( hi, hi, _MM_SHUFFLE( 3, 3, 2, 1 ) ) );
        const __m128 s = _mm_add_ps( lo, hi );
        _mm_storeu_ps( (float *)&center, _mm_move_ss( _mm_shuffle_ps( s, s, _MM_SHUFFLE( 2, 1, 0, 0 ) ), _mm_castsi128_ps( _mm_cvtsi32_si128( i ) ) ) );
#elif defined(__aarch64__) || defined(_M_ARM64)
        vst1q_f32( &box.min.x, vsetq_lane_f32( vgetq_lane_f32( hi, 0 ), lo, 3 ) );
        vst1_f32( &box.max.y, vget_low_f32( vextq_f32( hi, hi, 1 ) ) );
        const float32x4_t s = vaddq_f32( lo, hi );
        vst1q_f32( (float *)&center, vreinterpretq_f32_s32( vsetq_lane_s32( i, vreinterpretq_s32_f32( vextq_f32( s, s, 3 ) ), 0 ) ) );
#else
        wasm_v128_store( &box.min.x, wasm_i32x4_shuffle( lo, hi, 0, 1, 2, 4 ) );
        wasm_v128_store64_lane( &box.max.y, wasm_i32x4_shuffle( hi, hi, 1, 2, 1, 2 ), 0 );
        const v128_t s = wasm_f32x4_add( lo, hi );
        wasm_v128_store( &center, wasm_i32x4_replace_lane( wasm_i32x4_shuffle( s, s, 0, 0, 1, 2 ), 0, i ) );
#endif
        return;
    }
#endif
    box = insignificantlyExpandedTriBox( a, b, c );
    center = TreeCenter( i, box.min + box.max );
}

/// the boxes of 4 triangles given in SoA order as in AABBTree::SimdLeaf::coord, each the same as insignificantlyExpandedTriBox,
/// in SoA order as in AABBTree::SimdBoxes::box; empty (min = +inf, max = -inf) for the unused slots (NaN in the x and y of the first vertex)
inline void insignificantlyExpandedTriBoxes4( const float ( &coord )[3][3][4], float ( &box )[6][4] )
{
    constexpr float inf = std::numeric_limits<float>::infinity();
#if defined(__x86_64__) || defined(_M_X64)
    const __m128i one = _mm_set1_epi32( 1 );
    const __m128i absMask = _mm_set1_epi32( 0x7fffffff );
    const __m128i signBit = _mm_set1_epi32( int( 0x80000000u ) );
    const __m128i limit = _mm_set1_epi32( int( ( 0x7f7fffffu - 1 ) ^ 0x80000000u ) );
    const auto fast = [&]( __m128i x ) { return _mm_cmplt_epi32( _mm_xor_si128( _mm_sub_epi32( _mm_and_si128( x, absMask ), one ), signBit ), limit ); };
    const auto zero = [&]( __m128i x ) { return _mm_cmpeq_epi32( _mm_and_si128( x, absMask ), _mm_setzero_si128() ); };
    const __m128 c00 = _mm_load_ps( coord[0][0] ), c01 = _mm_load_ps( coord[0][1] );
    const __m128 used = _mm_or_ps( _mm_cmpeq_ps( c00, c00 ), _mm_cmpeq_ps( c01, c01 ) );
    __m128i ok = _mm_set1_epi32( -1 );
    __m128 lo[3], hi[3];
    for ( int c = 0; c < 3; ++c )
    {
        const __m128 a = _mm_load_ps( coord[0][c] ), b = _mm_load_ps( coord[1][c] ), cc = _mm_load_ps( coord[2][c] );
        // as in expandedTriBoxLanes, a NaN in b or cc is skipped as by Box::include, and a NaN in a is not fast
        const __m128i l = _mm_castps_si128( _mm_min_ps( cc, _mm_min_ps( b, a ) ) );
        const __m128i h = _mm_castps_si128( _mm_max_ps( cc, _mm_max_ps( b, a ) ) );
        const __m128i zl = zero( l ), zh = zero( h );
        ok = _mm_and_si128( ok, _mm_and_si128( _mm_or_si128( fast( l ), zl ), _mm_or_si128( fast( h ), zh ) ) );
        const __m128i l0 = _mm_or_si128( l, _mm_and_si128( zl, signBit ) );
        const __m128i h0 = _mm_andnot_si128( _mm_and_si128( zh, signBit ), h );
        lo[c] = _mm_castsi128_ps( _mm_sub_epi32( l0, _mm_or_si128( _mm_srai_epi32( l0, 31 ), one ) ) );
        hi[c] = _mm_castsi128_ps( _mm_add_epi32( h0, _mm_or_si128( _mm_srai_epi32( h0, 31 ), one ) ) );
    }
    if ( _mm_movemask_ps( _mm_or_ps( _mm_castsi128_ps( ok ), _mm_xor_ps( used, _mm_castsi128_ps( _mm_set1_epi32( -1 ) ) ) ) ) == 15 )
    {
        const __m128 pinf = _mm_set1_ps( inf ), ninf = _mm_set1_ps( -inf );
        for ( int c = 0; c < 3; ++c )
        {
            _mm_store_ps( box[c], _mm_or_ps( _mm_and_ps( used, lo[c] ), _mm_andnot_ps( used, pinf ) ) );
            _mm_store_ps( box[c + 3], _mm_or_ps( _mm_and_ps( used, hi[c] ), _mm_andnot_ps( used, ninf ) ) );
        }
        return;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    const uint32x4_t one = vdupq_n_u32( 1 );
    const uint32x4_t absMask = vdupq_n_u32( 0x7fffffff );
    const uint32x4_t limit = vdupq_n_u32( 0x7f7fffffu - 1 );
    const uint32x4_t signBit = vdupq_n_u32( 0x80000000u );
    const auto fast = [&]( uint32x4_t x ) { return vcltq_u32( vsubq_u32( vandq_u32( x, absMask ), one ), limit ); };
    const auto zero = [&]( uint32x4_t x ) { return vceqq_u32( vandq_u32( x, absMask ), vdupq_n_u32( 0 ) ); };
    const auto step = [&]( uint32x4_t x ) { return vorrq_u32( vreinterpretq_u32_s32( vshrq_n_s32( vreinterpretq_s32_u32( x ), 31 ) ), one ); };
    const float32x4_t c00 = vld1q_f32( coord[0][0] ), c01 = vld1q_f32( coord[0][1] );
    const uint32x4_t used = vorrq_u32( vceqq_f32( c00, c00 ), vceqq_f32( c01, c01 ) );
    uint32x4_t ok = vdupq_n_u32( 0xffffffffu );
    float32x4_t lo[3], hi[3];
    for ( int c = 0; c < 3; ++c )
    {
        const float32x4_t a = vld1q_f32( coord[0][c] ), b = vld1q_f32( coord[1][c] ), cc = vld1q_f32( coord[2][c] );
        const uint32x4_t notNan = vandq_u32( vandq_u32( vceqq_f32( a, a ), vceqq_f32( b, b ) ), vceqq_f32( cc, cc ) );
        const uint32x4_t l = vreinterpretq_u32_f32( vminq_f32( vminq_f32( a, b ), cc ) );
        const uint32x4_t h = vreinterpretq_u32_f32( vmaxq_f32( vmaxq_f32( a, b ), cc ) );
        const uint32x4_t zl = zero( l ), zh = zero( h );
        ok = vandq_u32( ok, vandq_u32( notNan, vandq_u32( vorrq_u32( fast( l ), zl ), vorrq_u32( fast( h ), zh ) ) ) );
        const uint32x4_t l0 = vorrq_u32( l, vandq_u32( zl, signBit ) );
        const uint32x4_t h0 = vbicq_u32( h, vandq_u32( zh, signBit ) );
        lo[c] = vreinterpretq_f32_u32( vsubq_u32( l0, step( l0 ) ) );
        hi[c] = vreinterpretq_f32_u32( vaddq_u32( h0, step( h0 ) ) );
    }
    if ( vminvq_u32( vorrq_u32( ok, vmvnq_u32( used ) ) ) != 0 )
    {
        const float32x4_t pinf = vdupq_n_f32( inf ), ninf = vdupq_n_f32( -inf );
        for ( int c = 0; c < 3; ++c )
        {
            vst1q_f32( box[c], vbslq_f32( used, lo[c], pinf ) );
            vst1q_f32( box[c + 3], vbslq_f32( used, hi[c], ninf ) );
        }
        return;
    }
#endif
    for ( int i = 0; i < 4; ++i )
    {
        if ( !( coord[0][0][i] == coord[0][0][i] ) && !( coord[0][1][i] == coord[0][1][i] ) )
        {
            for ( int c = 0; c < 3; ++c )
            {
                box[c][i] = inf;
                box[c + 3][i] = -inf;
            }
            continue;
        }
        const auto b = insignificantlyExpandedTriBox( { coord[0][0][i], coord[0][1][i], coord[0][2][i] },
            { coord[1][0][i], coord[1][1][i], coord[1][2][i] }, { coord[2][0][i], coord[2][1][i], coord[2][2][i] } );
        for ( int c = 0; c < 3; ++c )
        {
            box[c][i] = b.min[c];
            box[c + 3][i] = b.max[c];
        }
    }
}

/// the box of triangle k given in SoA order as in AABBTree::SimdLeaf::coord, the same as insignificantlyExpandedTriBox;
/// empty (min = +inf, max = -inf) for an unused slot (NaN in the x and y of the first vertex)
inline Box3f insignificantlyExpandedTriBox( const float ( &coord )[3][3][4], int k )
{
    if ( !( coord[0][0][k] == coord[0][0][k] ) && !( coord[0][1][k] == coord[0][1][k] ) )
    {
        constexpr float inf = std::numeric_limits<float>::infinity();
        return Box3f( Vector3f::diagonal( inf ), Vector3f::diagonal( -inf ) );
    }
    return insignificantlyExpandedTriBox( { coord[0][0][k], coord[0][1][k], coord[0][2][k] },
        { coord[1][0][k], coord[1][1][k], coord[1][2][k] }, { coord[2][0][k], coord[2][1][k], coord[2][2][k] } );
}

} //namespace MR

#undef MR_TRI_BOX_LANES
