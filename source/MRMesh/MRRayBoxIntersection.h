#pragma once
#include "MRBox.h"
#include "MRIntersectionPrecomputes.h"
#include "MRPch/MRBindingMacros.h"

namespace MR
{

/// \defgroup RayBoxIntersectionGroup Ray Box Intersection
/// \ingroup MathGroup
/// \{

template<typename T>
struct RayOrigin
{
    Vector3<T> p;
    RayOrigin( const Vector3<T> & ro ) : p( ro ) { }
};

/* CPU(X86_64) - AMD64 / Intel64 / x86_64 64-bit */
#if defined(__x86_64__) || defined(_M_X64)
template<>
struct RayOrigin<float>
{
    MR_BIND_IGNORE __m128 p;
    RayOrigin( const Vector3f & ro ) { p = _mm_set_ps( ro.x, ro.y, ro.z, 0 ); }
};

/// finds intersection between the Ray and the Box.
/// Precomputed values could be useful for several calls with the same direction,
/// see "An Efficient and Robust Ray-Box Intersection Algorithm" at https://people.csail.mit.edu/amy/papers/box-jgt.pdf
inline bool rayBoxIntersect( const Box3f& box, const RayOrigin<float> & rayOrigin, float & t0, float & t1, const IntersectionPrecomputes<float>& prec )
{
    __m128 l = _mm_set_ps( box.min.x, box.min.y, box.min.z, t0 );
    __m128 r = _mm_set_ps( box.max.x, box.max.y, box.max.z, t1 );
    l = _mm_sub_ps( l, rayOrigin.p );
    r = _mm_sub_ps( r, rayOrigin.p );
    l = _mm_mul_ps( l, prec.invDir );
    r = _mm_mul_ps( r, prec.invDir );

    __m128 a = _mm_min_ps( l, r );
    __m128 b = _mm_max_ps( l, r );

    __m128 aa = _mm_movehl_ps( a, a );
    aa = _mm_max_ps( aa, a );
    __m128 aaa = _mm_shuffle_ps( aa, aa, 1 );
    aaa = _mm_max_ss( aaa, aa );
    t0 = _mm_cvtss_f32( aaa );

    __m128 bb = _mm_movehl_ps( b, b );
    bb = _mm_min_ps( bb, b );
    __m128 bbb = _mm_shuffle_ps( bb, bb, 1 );
    bbb = _mm_min_ss( bbb, bb );
    t1 = _mm_cvtss_f32( bbb );

    return t0 <= t1;
}
#else
    #pragma message("rayBoxIntersect: no hardware optimized instructions")
#endif

template<typename T>
bool rayBoxIntersect( const Box3<T>& box, const RayOrigin<T> & rayOrigin, T & t0, T & t1, const IntersectionPrecomputes<T>& prec )
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

template <typename T = float>
bool rayBoxIntersect( const Box3<T>& box, const Line3<T>& line, T t0, T t1 )
{
    IntersectionPrecomputes<T> prec( line.d );
    return rayBoxIntersect( box, line, t0, t1, prec );
}

/// \}

}
