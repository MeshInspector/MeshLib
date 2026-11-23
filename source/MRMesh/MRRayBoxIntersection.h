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

#ifdef MR_SIMD4_HW
template<>
struct RayOrigin<float>
{
    MR_BIND_IGNORE Simd4::F p;
    RayOrigin( const Vector3f & ro ) : p( Simd4::make( ro.x, ro.y, ro.z, 0 ) ) { }
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

#ifdef MR_SIMD4_HW
    if constexpr ( std::is_same_v<T, float> )
    {
        using namespace Simd4;
        // both loads stay within the box, and the second one is the max corner shifted by one lane
        static_assert( sizeof( Box3f ) == 6 * sizeof( float ) );
        const float * const c = &box.min.x;
        const F l = mul( sub( loadu( c ), rayOrigin.p ), prec.invDir );
        const F r = mul( sub( shiftDown( loadu( c + 2 ) ), rayOrigin.p ), prec.invDir );
        // the 4-th lane holds garbage, so only the first three are reduced
        t0 = max3( minFast( l, r ), t0 );
        t1 = min3( maxFast( l, r ), t1 );
        return t0 <= t1;
    }
    else
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
