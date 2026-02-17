#pragma once
#include "MRBox.h"
#include "MRIntersectionPrecomputes.h"
#include "MRMesh/MRMacros.h"
#include "MRPch/MRBindingMacros.h"

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

// generic implementation of `rayBoxIntersect`
// see `rayBoxIntersect` for more info
template <typename T>
bool rayBoxIntersectGeneric( const Box3<T>& box, const RayOrigin<T> & rayOrigin, T & t0, T & t1, const IntersectionPrecomputes<T>& prec )
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

/// finds intersection between the Ray and the Box.
/// Precomputed values could be useful for several calls with the same direction,
/// see "An Efficient and Robust Ray-Box Intersection Algorithm" at https://people.csail.mit.edu/amy/papers/box-jgt.pdf
template <typename T>
bool rayBoxIntersect( const Box3<T>& box, const RayOrigin<T> & rayOrigin, T & t0, T & t1, const IntersectionPrecomputes<T>& prec )
{
    return rayBoxIntersectGeneric( box, rayOrigin, t0, t1, prec );
}

template <>
MRMESH_API bool rayBoxIntersect<float>( const Box3<float>& box, const RayOrigin<float>& rayOrigin, float& t0, float& t1, const IntersectionPrecomputes<float>& prec );

template <typename T = float>
bool rayBoxIntersect( const Box3<T>& box, const Line3<T>& line, T t0, T t1 )
{
    IntersectionPrecomputes<T> prec( line.d );
    return rayBoxIntersect( box, RayOrigin<T>( line.p ), t0, t1, prec );
}

/// \}

}
