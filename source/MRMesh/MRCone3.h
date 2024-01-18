#pragma once
#include "MRLine.h"

namespace MR
{

// Base class for cone parameterization

template <typename T>
class Cone3
{
public:
    Cone3()
    {}

    // inAxis -- apex position and main axis direction.  For any cone point dot( point , direction ) >=0
    // inAngle -- cone angle, main axis vs side
    // inHeight -- cone inHeight
    // main axis direction could be non normalized.
    Cone3( const Line3<T>& inAxis, T inAngle, T inHeight )
        :
        axis( inAxis ),
        angle( inAngle ),
        height( inHeight )
    {}

    // now we use an apex as center of the cone. 
    Vector3<T>& center( void )
    {
        return axis.p;
    }
    // now we use an apex as center of the cone. 
    const Vector3<T>& center( void ) const 
    {
        return axis.p;
    }
    // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
    Vector3<T>& direction( void )
    {
        return axis.d;
    }
    // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
    const Vector3<T>& direction( void ) const 
    {
        return axis.d;
    }
    // return cone apex position 
    Vector3<T>& apex( void )
    {
        return center();
    }
    // return cone apex position
    const Vector3<T>& apex( void ) const 
    {
        return center();
    }

    Line3<T> axis; // the combination of the apex of the cone and the direction of its main axis in space. 
    // for any cone point dot( point , direction ) >=0
    T angle = 0;       // cone angle, main axis vs side
    T height = 0;      // cone height
};



} // namespace MR
