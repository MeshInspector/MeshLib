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

    Cone3( const Line3<T>& inAxis, T inAngle, T inHeight )
        :
        axis( inAxis ),
        angle( inAngle ),
        height( inHeight )
    {}

    MR::Vector3<T>& center( void )
    {
        return axis.p;
    }

    const MR::Vector3<T>& center( void ) const 
    {
        return axis.p;
    }

    MR::Vector3<T>& direction( void )
    {
        return axis.d;
    }

    const MR::Vector3<T>& direction( void ) const 
    {
        return axis.d;
    }

    MR::Vector3<T>& apex( void )
    {
        return center();
    }

    const MR::Vector3<T>& apex( void ) const 
    {
        return center();
    }

    MR::Line3<T> axis; // the combination of the apex of the cone and the direction of its main axis in space. 
    // for any cone point dot( point , direction ) >=0
    T angle = 0;       // cone angle, main axis vs side
    T height = 0;      // cone height
};



} // namespace MR
