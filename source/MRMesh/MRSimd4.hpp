#pragma once

// minimal 4-lane float SIMD helpers for the traversals of AABBTree's SIMD layout:
// SSE2 on x86-64, NEON on ARM64, Wasm SIMD128, and a portable scalar fallback

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define MR_SIMD4_SSE
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define MR_SIMD4_NEON
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define MR_SIMD4_WASM
#endif

namespace MR::Simd4
{

#if defined(MR_SIMD4_SSE)

using F = __m128; ///< 4 floats
using M = __m128; ///< 4 lane masks
inline F load( const float* p ) { return _mm_load_ps( p ); }
inline F set1( float x ) { return _mm_set1_ps( x ); }
inline void store( float* p, F a ) { _mm_store_ps( p, a ); }
inline F add( F a, F b ) { return _mm_add_ps( a, b ); }
inline F sub( F a, F b ) { return _mm_sub_ps( a, b ); }
inline F mul( F a, F b ) { return _mm_mul_ps( a, b ); }
inline F div( F a, F b ) { return _mm_div_ps( a, b ); }
/// a < b ? a : b
inline F min( F a, F b ) { return _mm_min_ps( a, b ); }
/// a > b ? a : b
inline F max( F a, F b ) { return _mm_max_ps( a, b ); }
inline F abs( F a ) { return _mm_andnot_ps( _mm_set1_ps( -0.0f ), a ); }
inline M lt( F a, F b ) { return _mm_cmplt_ps( a, b ); }
inline M le( F a, F b ) { return _mm_cmple_ps( a, b ); }
inline M gt( F a, F b ) { return _mm_cmpgt_ps( a, b ); }
inline M ne( F a, F b ) { return _mm_cmpneq_ps( a, b ); }
inline M andm( M a, M b ) { return _mm_and_ps( a, b ); }
inline M orm( M a, M b ) { return _mm_or_ps( a, b ); }
inline M xorm( M a, M b ) { return _mm_xor_ps( a, b ); }
/// bit i is set if lane i of the mask is set
inline int bits( M m ) { return _mm_movemask_ps( m ); }

#elif defined(MR_SIMD4_NEON)

using F = float32x4_t;
using M = uint32x4_t;
inline F load( const float* p ) { return vld1q_f32( p ); }
inline F set1( float x ) { return vdupq_n_f32( x ); }
inline void store( float* p, F a ) { vst1q_f32( p, a ); }
inline F add( F a, F b ) { return vaddq_f32( a, b ); }
inline F sub( F a, F b ) { return vsubq_f32( a, b ); }
inline F mul( F a, F b ) { return vmulq_f32( a, b ); }
inline F div( F a, F b ) { return vdivq_f32( a, b ); }
inline F min( F a, F b ) { return vbslq_f32( vcltq_f32( a, b ), a, b ); }
inline F max( F a, F b ) { return vbslq_f32( vcgtq_f32( a, b ), a, b ); }
inline F abs( F a ) { return vabsq_f32( a ); }
inline M lt( F a, F b ) { return vcltq_f32( a, b ); }
inline M le( F a, F b ) { return vcleq_f32( a, b ); }
inline M gt( F a, F b ) { return vcgtq_f32( a, b ); }
inline M ne( F a, F b ) { return vmvnq_u32( vceqq_f32( a, b ) ); }
inline M andm( M a, M b ) { return vandq_u32( a, b ); }
inline M orm( M a, M b ) { return vorrq_u32( a, b ); }
inline M xorm( M a, M b ) { return veorq_u32( a, b ); }
inline int bits( M m )
{
    static const uint32_t w[4] = { 1, 2, 4, 8 };
    return int( vaddvq_u32( vandq_u32( m, vld1q_u32( w ) ) ) );
}

#elif defined(MR_SIMD4_WASM)

using F = v128_t;
using M = v128_t;
inline F load( const float* p ) { return wasm_v128_load( p ); }
inline F set1( float x ) { return wasm_f32x4_splat( x ); }
inline void store( float* p, F a ) { wasm_v128_store( p, a ); }
inline F add( F a, F b ) { return wasm_f32x4_add( a, b ); }
inline F sub( F a, F b ) { return wasm_f32x4_sub( a, b ); }
inline F mul( F a, F b ) { return wasm_f32x4_mul( a, b ); }
inline F div( F a, F b ) { return wasm_f32x4_div( a, b ); }
inline F min( F a, F b ) { return wasm_f32x4_pmin( b, a ); } // pmin( b, a ) = a < b ? a : b
inline F max( F a, F b ) { return wasm_f32x4_pmax( b, a ); } // pmax( b, a ) = b < a ? a : b
inline F abs( F a ) { return wasm_f32x4_abs( a ); }
inline M lt( F a, F b ) { return wasm_f32x4_lt( a, b ); }
inline M le( F a, F b ) { return wasm_f32x4_le( a, b ); }
inline M gt( F a, F b ) { return wasm_f32x4_gt( a, b ); }
inline M ne( F a, F b ) { return wasm_f32x4_ne( a, b ); }
inline M andm( M a, M b ) { return wasm_v128_and( a, b ); }
inline M orm( M a, M b ) { return wasm_v128_or( a, b ); }
inline M xorm( M a, M b ) { return wasm_v128_xor( a, b ); }
inline int bits( M m ) { return int( wasm_i32x4_bitmask( m ) ); }

#else

struct F { float v[4]; };
struct M { bool v[4]; };
inline F load( const float* p ) { return { { p[0], p[1], p[2], p[3] } }; }
inline F set1( float x ) { return { { x, x, x, x } }; }
inline void store( float* p, F a ) { for ( int i = 0; i < 4; ++i ) p[i] = a.v[i]; }
#define MR_SIMD4_LANES( R, expr ) R r; for ( int i = 0; i < 4; ++i ) r.v[i] = ( expr ); return r
inline F add( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] + b.v[i] ); }
inline F sub( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] - b.v[i] ); }
inline F mul( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] * b.v[i] ); }
inline F div( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] / b.v[i] ); }
inline F min( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] < b.v[i] ? a.v[i] : b.v[i] ); }
inline F max( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] > b.v[i] ? a.v[i] : b.v[i] ); }
inline F abs( F a ) { MR_SIMD4_LANES( F, a.v[i] < 0 ? -a.v[i] : a.v[i] ); }
inline M lt( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] < b.v[i] ); }
inline M le( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] <= b.v[i] ); }
inline M gt( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] > b.v[i] ); }
inline M ne( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] != b.v[i] ); }
inline M andm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] && b.v[i] ); }
inline M orm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] || b.v[i] ); }
inline M xorm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] != b.v[i] ); }
#undef MR_SIMD4_LANES
inline int bits( M m ) { return int( m.v[0] ) | int( m.v[1] ) << 1 | int( m.v[2] ) << 2 | int( m.v[3] ) << 3; }

#endif

/// copies src to 16-byte aligned dst of a size multiple of 16 bytes, bypassing the caches where possible (non-temporal stores):
/// for large arrays written once, it saves reading the destination lines first; call streamFence() before other threads read them
template <typename T>
inline void streamCopy( T & dst, const T & src )
{
    static_assert( sizeof( T ) % 16 == 0 && alignof( T ) >= 16 );
#if defined(MR_SIMD4_SSE)
    const float * s = reinterpret_cast<const float *>( &src );
    float * d = reinterpret_cast<float *>( &dst );
    for ( size_t i = 0; i < sizeof( T ) / 4; i += 4 )
        _mm_stream_ps( d + i, _mm_load_ps( s + i ) );
#else
    dst = src;
#endif
}

/// makes the non-temporal stores of this thread visible to other threads
inline void streamFence()
{
#if defined(MR_SIMD4_SSE)
    _mm_sfence();
#endif
}

/// the index of the lowest set bit of a non-zero 4-bit mask
inline int lowestBit( int m ) { return ( m & 1 ) ? 0 : ( m & 2 ) ? 1 : ( m & 4 ) ? 2 : 3; }

} // namespace MR::Simd4
