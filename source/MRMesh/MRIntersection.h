#pragma once

#include "MRPlane3.h"
#include "MRLine3.h"
#include <optional>

namespace MR
{

// finds an intersection between a plane and a line;
// returns nullopt if they are parallel
template<typename T>
std::optional<Vector3<T>> intersection( const Plane3<T> & plane, const Line3<T> & line )
{
    const auto den = dot( plane.n, line.d );
    if ( den == 0 )
        return {};
    return line.p + ( plane.d - dot( plane.n, line.p ) ) / den * line.d;
}

} //namespace MR
