#pragma once

// minimal 4-lane float SIMD helpers for the traversals of AABBTree's SIMD layout and rayBoxIntersect:
// SSE2 on x86-64, NEON on ARM64, Wasm SIMD128, and a portable scalar fallback

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define MR_SIMD4_SSE
#define MR_SIMD4_HW
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define MR_SIMD4_NEON
#define MR_SIMD4_HW
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define MR_SIMD4_WASM
#define MR_SIMD4_HW
#endif

namespace MR::Simd4
{

#if defined(MR_SIMD4_SSE)

using F = __m128; ///< 4 floats
using M = __m128; ///< 4 lane masks
inline F load( const float* p ) { return _mm_load_ps( p ); }
/// p need not be 16-byte aligned
inline F loadu( const float* p ) { return _mm_loadu_ps( p ); }
inline F set1( float x ) { return _mm_set1_ps( x ); }
/// lanes a, b, c, d
inline F make( float a, float b, float c, float d ) { return _mm_setr_ps( a, b, c, d ); }
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
/// the same as min / max if no lane is NaN, except possibly the sign of a zero result
inline F minFast( F a, F b ) { return min( a, b ); }
inline F maxFast( F a, F b ) { return max( a, b ); }
inline M lt( F a, F b ) { return _mm_cmplt_ps( a, b ); }
inline M le( F a, F b ) { return _mm_cmple_ps( a, b ); }
inline M gt( F a, F b ) { return _mm_cmpgt_ps( a, b ); }
inline M ne( F a, F b ) { return _mm_cmpneq_ps( a, b ); }
inline M andm( M a, M b ) { return _mm_and_ps( a, b ); }
inline M orm( M a, M b ) { return _mm_or_ps( a, b ); }
inline M xorm( M a, M b ) { return _mm_xor_ps( a, b ); }
/// bit i is set if lane i of the mask is set
inline int bits( M m ) { return _mm_movemask_ps( m ); }
/// m ? a : b in each lane
inline F select( M m, F a, F b ) { return _mm_or_ps( _mm_and_ps( m, a ), _mm_andnot_ps( m, b ) ); }
/// lanes 1, 2, 3 of a in lanes 0, 1, 2; lane 3 is unspecified
inline F shiftDown( F a ) { return _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 3, 2, 1 ) ); }
/// the maximum of x and lanes 0, 1, 2 of a
inline float max3( F a, float x )
{
    const F az = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 2, 2, 2 ) ), ay = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 1, 1, 1, 1 ) );
    return _mm_cvtss_f32( _mm_max_ss( _mm_max_ss( _mm_max_ss( a, az ), ay ), _mm_set_ss( x ) ) );
}
/// the minimum of x and lanes 0, 1, 2 of a
inline float min3( F a, float x )
{
    const F az = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 2, 2, 2 ) ), ay = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 1, 1, 1, 1 ) );
    return _mm_cvtss_f32( _mm_min_ss( _mm_min_ss( _mm_min_ss( a, az ), ay ), _mm_set_ss( x ) ) );
}

#elif defined(MR_SIMD4_NEON)

using F = float32x4_t;
using M = uint32x4_t;
inline F load( const float* p ) { return vld1q_f32( p ); }
inline F loadu( const float* p ) { return vld1q_f32( p ); }
inline F set1( float x ) { return vdupq_n_f32( x ); }
inline F make( float a, float b, float c, float d ) { const float v[4] = { a, b, c, d }; return vld1q_f32( v ); }
inline void store( float* p, F a ) { vst1q_f32( p, a ); }
inline F add( F a, F b ) { return vaddq_f32( a, b ); }
inline F sub( F a, F b ) { return vsubq_f32( a, b ); }
inline F mul( F a, F b ) { return vmulq_f32( a, b ); }
inline F div( F a, F b ) { return vdivq_f32( a, b ); }
inline F min( F a, F b ) { return vbslq_f32( vcltq_f32( a, b ), a, b ); }
inline F max( F a, F b ) { return vbslq_f32( vcgtq_f32( a, b ), a, b ); }
inline F abs( F a ) { return vabsq_f32( a ); }
/// the same as min / max if no lane is NaN, except possibly the sign of a zero result; one instruction instead of two
inline F minFast( F a, F b ) { return vminq_f32( a, b ); }
inline F maxFast( F a, F b ) { return vmaxq_f32( a, b ); }
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
inline F select( M m, F a, F b ) { return vbslq_f32( m, a, b ); }
inline F shiftDown( F a ) { return vextq_f32( a, a, 1 ); }
inline float max3( F a, float x ) { return vmaxvq_f32( vsetq_lane_f32( x, a, 3 ) ); }
inline float min3( F a, float x ) { return vminvq_f32( vsetq_lane_f32( x, a, 3 ) ); }

#elif defined(MR_SIMD4_WASM)

using F = v128_t;
using M = v128_t;
inline F load( const float* p ) { return wasm_v128_load( p ); }
inline F loadu( const float* p ) { return wasm_v128_load( p ); }
inline F set1( float x ) { return wasm_f32x4_splat( x ); }
inline F make( float a, float b, float c, float d ) { return wasm_f32x4_make( a, b, c, d ); }
inline void store( float* p, F a ) { wasm_v128_store( p, a ); }
inline F add( F a, F b ) { return wasm_f32x4_add( a, b ); }
inline F sub( F a, F b ) { return wasm_f32x4_sub( a, b ); }
inline F mul( F a, F b ) { return wasm_f32x4_mul( a, b ); }
inline F div( F a, F b ) { return wasm_f32x4_div( a, b ); }
inline F min( F a, F b ) { return wasm_f32x4_pmin( b, a ); } // pmin( b, a ) = a < b ? a : b
inline F max( F a, F b ) { return wasm_f32x4_pmax( b, a ); } // pmax( b, a ) = b < a ? a : b
inline F abs( F a ) { return wasm_f32x4_abs( a ); }
/// the same as min / max if no lane is NaN, except possibly the sign of a zero result
inline F minFast( F a, F b ) { return min( a, b ); }
inline F maxFast( F a, F b ) { return max( a, b ); }
inline M lt( F a, F b ) { return wasm_f32x4_lt( a, b ); }
inline M le( F a, F b ) { return wasm_f32x4_le( a, b ); }
inline M gt( F a, F b ) { return wasm_f32x4_gt( a, b ); }
inline M ne( F a, F b ) { return wasm_f32x4_ne( a, b ); }
inline M andm( M a, M b ) { return wasm_v128_and( a, b ); }
inline M orm( M a, M b ) { return wasm_v128_or( a, b ); }
inline M xorm( M a, M b ) { return wasm_v128_xor( a, b ); }
inline int bits( M m ) { return int( wasm_i32x4_bitmask( m ) ); }
inline F select( M m, F a, F b ) { return wasm_v128_bitselect( a, b, m ); }
inline F shiftDown( F a ) { return wasm_i32x4_shuffle( a, a, 1, 2, 3, 3 ); }
// Wasm has neither a cross-lane reduction nor an operation on the first lane alone
inline float max3( F a, float x )
{
    const F m = wasm_f32x4_pmax( wasm_f32x4_pmax( a, wasm_i32x4_shuffle( a, a, 2, 2, 2, 2 ) ), wasm_i32x4_shuffle( a, a, 1, 1, 1, 1 ) );
    const float y = wasm_f32x4_extract_lane( m, 0 );
    return x < y ? y : x;
}
inline float min3( F a, float x )
{
    const F m = wasm_f32x4_pmin( wasm_f32x4_pmin( a, wasm_i32x4_shuffle( a, a, 2, 2, 2, 2 ) ), wasm_i32x4_shuffle( a, a, 1, 1, 1, 1 ) );
    const float y = wasm_f32x4_extract_lane( m, 0 );
    return y < x ? y : x;
}

#else

struct F { float v[4]; };
struct M { bool v[4]; };
inline F load( const float* p ) { return { { p[0], p[1], p[2], p[3] } }; }
inline F loadu( const float* p ) { return load( p ); }
inline F set1( float x ) { return { { x, x, x, x } }; }
inline F make( float a, float b, float c, float d ) { return { { a, b, c, d } }; }
inline void store( float* p, F a ) { for ( int i = 0; i < 4; ++i ) p[i] = a.v[i]; }
#define MR_SIMD4_LANES( R, expr ) R r; for ( int i = 0; i < 4; ++i ) r.v[i] = ( expr ); return r
inline F add( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] + b.v[i] ); }
inline F sub( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] - b.v[i] ); }
inline F mul( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] * b.v[i] ); }
inline F div( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] / b.v[i] ); }
inline F min( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] < b.v[i] ? a.v[i] : b.v[i] ); }
inline F max( F a, F b ) { MR_SIMD4_LANES( F, a.v[i] > b.v[i] ? a.v[i] : b.v[i] ); }
inline F abs( F a ) { MR_SIMD4_LANES( F, a.v[i] < 0 ? -a.v[i] : a.v[i] ); }
inline F minFast( F a, F b ) { return min( a, b ); }
inline F maxFast( F a, F b ) { return max( a, b ); }
inline M lt( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] < b.v[i] ); }
inline M le( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] <= b.v[i] ); }
inline M gt( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] > b.v[i] ); }
inline M ne( F a, F b ) { MR_SIMD4_LANES( M, a.v[i] != b.v[i] ); }
inline M andm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] && b.v[i] ); }
inline M orm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] || b.v[i] ); }
inline M xorm( M a, M b ) { MR_SIMD4_LANES( M, a.v[i] != b.v[i] ); }
inline F select( M m, F a, F b ) { MR_SIMD4_LANES( F, m.v[i] ? a.v[i] : b.v[i] ); }
#undef MR_SIMD4_LANES
inline int bits( M m ) { return int( m.v[0] ) | int( m.v[1] ) << 1 | int( m.v[2] ) << 2 | int( m.v[3] ) << 3; }
inline F shiftDown( F a ) { return { { a.v[1], a.v[2], a.v[3], a.v[3] } }; }
inline float max3( F a, float x ) { for ( int i = 0; i < 3; ++i ) x = x < a.v[i] ? a.v[i] : x; return x; }
inline float min3( F a, float x ) { for ( int i = 0; i < 3; ++i ) x = a.v[i] < x ? a.v[i] : x; return x; }

#endif

/// whether streamCopy and streamCopyBytes use non-temporal stores; otherwise it is better to write the destination in place
#if defined(MR_SIMD4_SSE) || ( defined(MR_SIMD4_NEON) && defined(__clang__) )
inline constexpr bool cStreamStores = true;
#else
inline constexpr bool cStreamStores = false;
#endif

/// copies size bytes (a multiple of 16) to 16-byte aligned dst, bypassing the caches where possible (non-temporal stores)
inline void streamCopyBytes( void * dst, const void * src, std::size_t size )
{
#if defined(MR_SIMD4_SSE)
    for ( std::size_t i = 0; i < size; i += 16 )
        _mm_stream_si128( reinterpret_cast<__m128i *>( static_cast<char *>( dst ) + i ),
            _mm_loadu_si128( reinterpret_cast<const __m128i *>( static_cast<const char *>( src ) + i ) ) );
#elif defined(MR_SIMD4_NEON) && defined(__clang__)
    // stnp of two 16-byte registers, which only clang gives a way to ask for
    typedef std::uint8_t V32 __attribute__(( vector_size( 32 ), aligned( 16 ) ));
    std::size_t i = 0;
    for ( ; i + 32 <= size; i += 32 )
    {
        V32 v;
        std::memcpy( &v, static_cast<const char *>( src ) + i, 32 );
        __builtin_nontemporal_store( v, reinterpret_cast<V32 *>( static_cast<char *>( dst ) + i ) );
    }
    for ( ; i < size; i += 16 )
        __builtin_nontemporal_store( vld1q_u8( static_cast<const std::uint8_t *>( src ) + i ),
            reinterpret_cast<uint8x16_t *>( static_cast<char *>( dst ) + i ) );
#else
    std::memcpy( dst, src, size );
#endif
}

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
#elif defined(MR_SIMD4_NEON) && defined(__clang__)
    streamCopyBytes( &dst, &src, sizeof( T ) );
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

/// squared distances from point (px, py, pz) to 4 boxes given in SoA order as in AABBTree::SimdNode::box,
/// computed exactly as Box3f::getDistanceSq( pt ); infinite for empty boxes
inline F boxDistSq( const float ( &box )[6][4], F px, F py, F pz )
{
    const F p[3] = { px, py, pz };
    F res = set1( 0.0f );
    for ( int c = 0; c < 3; ++c )
    {
        const F lo = load( box[c] ), hi = load( box[c + 3] );
        const F dlo = sub( p[c], lo ), dhi = sub( p[c], hi );
        res = add( res, select( lt( p[c], lo ), mul( dlo, dlo ), select( gt( p[c], hi ), mul( dhi, dhi ), set1( 0.0f ) ) ) );
    }
    return res;
}

} // namespace MR::Simd4
