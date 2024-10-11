#pragma once

#include "MRMeshFwd.h"

#include <limits>

namespace MR
{

/// \brief encodes a point inside a line segment using relative distance in [0,1]
/// \ingroup MathGroup
template <typename T>
struct SegmPoint
{
    T a = 0; ///< a in [0,1], a=0 => point is in v0, a=1 => point is in v1

    static constexpr auto eps = 10 * std::numeric_limits<T>::epsilon();

    SegmPoint() = default;
    SegmPoint( T a ) : a( a ) { }
    operator T() const { return a; }
    operator T&() { return a; }

    /// given values in two vertices, computes interpolated value at this point
    template <typename U>
    [[nodiscard]] U interpolate( const U & v0, const U & v1 ) const
    {
        return ( 1 - a ) * v0 + a * v1;
    }

    /// returns [0,1] if the point is in a vertex or -1 otherwise
    [[nodiscard]] int inVertex() const
    {
        if ( a <= eps )
            return 0;
        if ( 1 - a <= eps )
            return 1;
        return -1;
    }

    /// represents the same point relative to oppositely directed segment
    [[nodiscard]] SegmPoint sym() const { return { 1 - a }; }
    /// returns true if two points have equal (a) representation
    [[nodiscard]] bool operator==( const SegmPoint& rhs ) const = default;
    [[nodiscard]] bool operator==( T rhs ) const { return a == rhs; }
};

} // namespace MR
