#pragma once

#include "MRBox.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace MR
{

/// returns the same box as the empty box including a, b and c, insignificantly expanded, by SIMD instructions where available:
/// if |x| < max in all coordinates, then nextDown and nextUp of nonzero x differ from it by one in the bits,
/// and of +-0 are -denorm_min and +denorm_min, which the steps below give for -0 in lo and +0 in hi
inline Box3f insignificantlyExpandedTriBox( const Vector3f & a, const Vector3f & b, const Vector3f & c )
{
#if defined(__x86_64__) || defined(_M_X64)
    // the lanes of a point: x, y, z, z
    const auto load = []( const Vector3f & p ) { return _mm_movelh_ps( _mm_castsi128_ps( _mm_loadl_epi64( (const __m128i *)&p.x ) ), _mm_load1_ps( &p.z ) ); };
    const __m128 va = load( a ), vb = load( b ), vc = load( c );
    // a NaN in b or c is skipped as by Box::include, and a NaN in a leads to the scalar code below
    const __m128i lo = _mm_castps_si128( _mm_min_ps( vc, _mm_min_ps( vb, va ) ) );
    const __m128i hi = _mm_castps_si128( _mm_max_ps( vc, _mm_max_ps( vb, va ) ) );
    const __m128i one = _mm_set1_epi32( 1 );
    const __m128i absMask = _mm_set1_epi32( 0x7fffffff );
    const __m128i signBit = _mm_set1_epi32( int( 0x80000000u ) ); // to compare unsigned integers as signed ones
    const __m128i limit = _mm_set1_epi32( int( ( 0x7f7fffffu - 1 ) ^ 0x80000000u ) );
    const auto fast = [&]( __m128i x ) { return _mm_cmplt_epi32( _mm_xor_si128( _mm_sub_epi32( _mm_and_si128( x, absMask ), one ), signBit ), limit ); };
    const auto zero = [&]( __m128i x ) { return _mm_cmpeq_epi32( _mm_and_si128( x, absMask ), _mm_setzero_si128() ); };
    const __m128i zLo = zero( lo ), zHi = zero( hi );
    if ( _mm_movemask_ps( _mm_castsi128_ps( _mm_and_si128( _mm_or_si128( fast( lo ), zLo ), _mm_or_si128( fast( hi ), zHi ) ) ) ) == 15 )
    {
        const __m128i lo0 = _mm_or_si128( lo, _mm_and_si128( zLo, signBit ) );
        const __m128i hi0 = _mm_andnot_si128( _mm_and_si128( zHi, signBit ), hi );
        // minus one for positive and plus one for negative lo, plus one for positive and minus one for negative hi
        alignas( 16 ) float l[4], h[4];
        _mm_store_ps( l, _mm_castsi128_ps( _mm_sub_epi32( lo0, _mm_or_si128( _mm_srai_epi32( lo0, 31 ), one ) ) ) );
        _mm_store_ps( h, _mm_castsi128_ps( _mm_add_epi32( hi0, _mm_or_si128( _mm_srai_epi32( hi0, 31 ), one ) ) ) );
        return Box3f( Vector3f( l[0], l[1], l[2] ), Vector3f( h[0], h[1], h[2] ) );
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    const auto load = []( const Vector3f & p ) { return vcombine_f32( vld1_f32( &p.x ), vld1_dup_f32( &p.z ) ); };
    const float32x4_t va = load( a ), vb = load( b ), vc = load( c );
    // any NaN, quiet or signaling, in a, b or c leads to the scalar code below, tested on the points themselves
    const uint32x4_t notNan = vandq_u32( vandq_u32( vceqq_f32( va, va ), vceqq_f32( vb, vb ) ), vceqq_f32( vc, vc ) );
    const uint32x4_t lo = vreinterpretq_u32_f32( vminq_f32( vminq_f32( va, vb ), vc ) );
    const uint32x4_t hi = vreinterpretq_u32_f32( vmaxq_f32( vmaxq_f32( va, vb ), vc ) );
    const uint32x4_t one = vdupq_n_u32( 1 );
    const uint32x4_t absMask = vdupq_n_u32( 0x7fffffff );
    const uint32x4_t limit = vdupq_n_u32( 0x7f7fffffu - 1 );
    const auto fast = [&]( uint32x4_t x ) { return vcltq_u32( vsubq_u32( vandq_u32( x, absMask ), one ), limit ); };
    const auto zero = [&]( uint32x4_t x ) { return vceqq_u32( vandq_u32( x, absMask ), vdupq_n_u32( 0 ) ); };
    const uint32x4_t zLo = zero( lo ), zHi = zero( hi );
    if ( vminvq_u32( vandq_u32( notNan, vandq_u32( vorrq_u32( fast( lo ), zLo ), vorrq_u32( fast( hi ), zHi ) ) ) ) != 0 )
    {
        const uint32x4_t signBit = vdupq_n_u32( 0x80000000u );
        const uint32x4_t lo0 = vorrq_u32( lo, vandq_u32( zLo, signBit ) );
        const uint32x4_t hi0 = vbicq_u32( hi, vandq_u32( zHi, signBit ) );
        const auto step = [&]( uint32x4_t x ) { return vorrq_u32( vreinterpretq_u32_s32( vshrq_n_s32( vreinterpretq_s32_u32( x ), 31 ) ), one ); };
        float l[4], h[4];
        vst1q_f32( l, vreinterpretq_f32_u32( vsubq_u32( lo0, step( lo0 ) ) ) );
        vst1q_f32( h, vreinterpretq_f32_u32( vaddq_u32( hi0, step( hi0 ) ) ) );
        return Box3f( Vector3f( l[0], l[1], l[2] ), Vector3f( h[0], h[1], h[2] ) );
    }
#elif defined(__wasm_simd128__)
    const auto load = []( const Vector3f & p ) { return wasm_i32x4_shuffle( wasm_v128_load64_zero( &p.x ), wasm_v128_load32_splat( &p.z ), 0, 1, 4, 5 ); };
    const v128_t va = load( a ), vb = load( b ), vc = load( c );
    // pmin( x, v ) = v < x ? v : x skips a NaN in v as Box::include does, and a NaN in a leads to the scalar code below
    const v128_t lo = wasm_f32x4_pmin( wasm_f32x4_pmin( va, vb ), vc );
    const v128_t hi = wasm_f32x4_pmax( wasm_f32x4_pmax( va, vb ), vc );
    const v128_t one = wasm_i32x4_splat( 1 );
    const v128_t absMask = wasm_i32x4_splat( 0x7fffffff );
    const v128_t limit = wasm_i32x4_splat( int( 0x7f7fffffu - 1 ) );
    const auto fast = [&]( v128_t x ) { return wasm_u32x4_lt( wasm_i32x4_sub( wasm_v128_and( x, absMask ), one ), limit ); };
    const auto zero = [&]( v128_t x ) { return wasm_i32x4_eq( wasm_v128_and( x, absMask ), wasm_i32x4_splat( 0 ) ); };
    const v128_t zLo = zero( lo ), zHi = zero( hi );
    if ( wasm_i32x4_all_true( wasm_v128_and( wasm_v128_or( fast( lo ), zLo ), wasm_v128_or( fast( hi ), zHi ) ) ) )
    {
        const v128_t signBit = wasm_i32x4_splat( int( 0x80000000u ) );
        const v128_t lo0 = wasm_v128_or( lo, wasm_v128_and( zLo, signBit ) );
        const v128_t hi0 = wasm_v128_andnot( hi, wasm_v128_and( zHi, signBit ) );
        const auto step = [&]( v128_t x ) { return wasm_v128_or( wasm_i32x4_shr( x, 31 ), one ); };
        alignas( 16 ) float l[4], h[4];
        wasm_v128_store( l, wasm_i32x4_sub( lo0, step( lo0 ) ) );
        wasm_v128_store( h, wasm_i32x4_add( hi0, step( hi0 ) ) );
        return Box3f( Vector3f( l[0], l[1], l[2] ), Vector3f( h[0], h[1], h[2] ) );
    }
#endif
    Box3f box;
    box.include( a );
    box.include( b );
    box.include( c );
    return box.insignificantlyExpanded();
}

} //namespace MR
