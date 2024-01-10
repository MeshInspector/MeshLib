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

    Cone3( const Line3<T>& inAxis, T inAngle, T inLength )
        :
        position( inAxis ),
        angle( inAngle ),
        length( inLength )
    {}

    inline MR::Vector3<T>& center( void )
    {
        return position.p;
    }
    inline MR::Vector3<T>& direction( void )
    {
        return position.d;
    }
    inline MR::Vector3<T>& apex( void )
    {
        return center();
    }

    MR::Line3<T> position; // the combination of the apex of the cone and the direction of its main axis in space
    T angle = 0; // cone angle
    T length = 0; // cone length
};



} // namespace MR
