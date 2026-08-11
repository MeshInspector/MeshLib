#pragma once
#include "MRVector3.h"

namespace MR
{
 enum class Axis
 {
     X,
     Y,
     Z,
     Count
 };

/// one of the six signed coordinate axes; the order lets axis() and isNegative() be index arithmetic
enum class SignedAxis
{
    PlusX,
    PlusY,
    PlusZ,
    MinusX,
    MinusY,
    MinusZ,
    Count
};

/// the coordinate axis (a) is directed along
[[nodiscard]] inline Axis axis( SignedAxis a ) { return Axis( int( a ) % 3 ); }

/// whether (a) points against its coordinate axis
[[nodiscard]] inline bool isNegative( SignedAxis a ) { return int( a ) >= 3; }

/// the signed axis (v) is most aligned with: its largest-magnitude component, and that component's sign
template <typename T>
[[nodiscard]] SignedAxis dominantSignedAxis( const Vector3<T>& v )
{
    int d = 0;
    for ( int i = 1; i < 3; ++i )
        if ( v[i] * v[i] > v[d] * v[d] )
            d = i;
    return SignedAxis( v[d] < 0 ? d + 3 : d );
}
}
