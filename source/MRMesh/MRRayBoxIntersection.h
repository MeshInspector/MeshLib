#pragma once
#include "MRBox.h"
#include "MRIntersectionPrecomputes.h"
#include "MRMesh/MRMacros.h"
#include "MRPch/MRBindingMacros.h"
#include <cassert>

namespace MR
{

/// \defgroup RayBoxIntersectionGroup Ray Box Intersection
/// \ingroup MathGroup
/// \{

template<typename T>
struct RayOrigin
{
    // This is hidden to match the specialization below.
    MR_BIND_IGNORE Vector3<T> p;
    RayOrigin( const Vector3<T> & ro ) : p( ro ) { }
};

/* CPU(X86_64) - AMD64 / Intel64 / x86_64 64-bit */
#if defined(__x86_64__) || defined(_M_X64)
template<>
struct RayOrigin<float>
{
    MR_BIND_IGNORE __m128 p;
    RayOrigin( const Vector3f & ro ) { p = _mm_set_ps( 0, ro.z, ro.y, ro.x ); }
};

/* CPU(ARM64) - AArch64 */
#elif defined(__aarch64__) || defined(_M_ARM64)
template<>
struct RayOrigin<float>
{
    MR_BIND_IGNORE float32x4_t p;
    RayOrigin( const Vector3f & ro ) { p = toFloat32x4( ro.x, ro.y, ro.z, 0 ); }
};
#endif

/// finds intersection between the Ray and the Box.
/// The box must be valid, otherwise the result is undefined.
/// Precomputed values could be useful for several calls with the same direction,
/// see "An Efficient and Robust Ray-Box Intersection Algorithm" at https://people.csail.mit.edu/amy/papers/box-jgt.pdf
template <typename T = float>
bool rayBoxIntersect( const Box3<T>& box, const RayOrigin<T> & rayOrigin, T & t0, T & t1, const IntersectionPrecomputes<T>& prec )
{
    assert( box.valid() );

    #if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>)
    {
        // both loads stay within the box, and the second one is the max corner shifted by one lane
        static_assert( sizeof( Box3f ) == 6 * sizeof( float ) );
        const float * const c = &box.min.x;
        __m128 l = _mm_loadu_ps( c );
        __m128 h = _mm_loadu_ps( c + 2 );
        __m128 r = _mm_shuffle_ps( h, h, _MM_SHUFFLE( 3, 3, 2, 1 ) );

        l = _mm_mul_ps( _mm_sub_ps( l, rayOrigin.p ), prec.invDir );
        r = _mm_mul_ps( _mm_sub_ps( r, rayOrigin.p ), prec.invDir );

        __m128 a = _mm_min_ps( l, r );
        __m128 b = _mm_max_ps( l, r );

        // the 4-th lane holds garbage, so only the first three are reduced
        __m128 az = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 2, 2, 2 ) );
        __m128 ay = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 1, 1, 1, 1 ) );
        t0 = _mm_cvtss_f32( _mm_max_ss( _mm_max_ss( _mm_max_ss( a, az ), ay ), _mm_load_ss( &t0 ) ) );

        __m128 bz = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 2, 2, 2, 2 ) );
        __m128 by = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 1, 1, 1, 1 ) );
        t1 = _mm_cvtss_f32( _mm_min_ss( _mm_min_ss( _mm_min_ss( b, bz ), by ), _mm_load_ss( &t1 ) ) );

        return t0 <= t1;
    }
    else
    #elif defined(__aarch64__) || defined(_M_ARM64)
    if constexpr (std::is_same_v<T, float>)
    {
        // both loads stay within the box, and the second one is the max corner shifted by one lane;
        // the 4-th lane carries t0/t1 itself, since the origin is 0 and the inverted direction is 1 there
        static_assert( sizeof( Box3f ) == 6 * sizeof( float ) );
        const float * const c = &box.min.x;
        float32x4_t h = vld1q_f32( c + 2 );
        float32x4_t l = vsetq_lane_f32( t0, vld1q_f32( c ), 3 );
        float32x4_t r = vsetq_lane_f32( t1, vextq_f32( h, h, 1 ), 3 );

        l = vmulq_f32( vsubq_f32( l, rayOrigin.p ), prec.invDir );
        r = vmulq_f32( vsubq_f32( r, rayOrigin.p ), prec.invDir );

        t0 = vmaxvq_f32( vminq_f32( l, r ) );
        t1 = vminvq_f32( vmaxq_f32( l, r ) );

        return t0 <= t1;
    }
    else
    #else
    #pragma message("rayBoxIntersect: no hardware optimized instructions")
    #endif
    {
        const Vector3i& sign = prec.sign;

        // compare and update x-dimension with t0-t1
        t1 = std::min( (box[sign.x].x - rayOrigin.p.x) * prec.invDir.x, t1 );
        t0 = std::max( (box[1 - sign.x].x - rayOrigin.p.x) * prec.invDir.x, t0 );

        // compare and update y-dimension with t0-t1
        t1 = std::min( (box[sign.y].y - rayOrigin.p.y) * prec.invDir.y, t1 );
        t0 = std::max( (box[1 - sign.y].y - rayOrigin.p.y) * prec.invDir.y, t0 );

        // compare and update z-dimension with t0-t1
        t1 = std::min( (box[sign.z].z - rayOrigin.p.z) * prec.invDir.z, t1 );
        t0 = std::max( (box[1 - sign.z].z - rayOrigin.p.z) * prec.invDir.z, t0 );
        return t0 <= t1;
    }
}

/// finds intersection between the Ray and the Box, which must be valid
template <typename T = float>
bool rayBoxIntersect( const Box3<T>& box, const Line3<T>& line, T t0, T t1 )
{
    IntersectionPrecomputes<T> prec( line.d );
    return rayBoxIntersect( box, RayOrigin<T>( line.p ), t0, t1, prec );
}

/// \}

}
