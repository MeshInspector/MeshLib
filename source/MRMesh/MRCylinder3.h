#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRLine.h"

namespace MR
{
// A class describing a cylinder as a mathematical object.A cylinder is represented by a centerline, a radius, and a length.template <typename T>
// TODO: Cylinder3 could be infinite. For example for infinite Cylinder3 we could use negative length or length = -1
template <typename T>
class Cylinder3
{
public:
    Cylinder3()
    {}

    Cylinder3( const Vector3<T>& inCenter, const Vector3<T>& inDirectoin, T inRadius, T inLength )
        :
        mainAxis( inCenter, inDirectoin ),
        radius( inRadius ),
        length( inLength )
    {}
    Cylinder3( const Line3<T>& inAxis, T inRadius, T inLength )
        :
        mainAxis( inAxis ),
        radius( inRadius ),
        length( inLength )
    {}

    Vector3<T>& center( void )
    {
        return mainAxis.p;
    }

    const Vector3<T>& center( void ) const
    {
        return mainAxis.p;
    }

    Vector3<T>& direction( void )
    {
        return mainAxis.d;
    }

    const Vector3<T>& direction( void ) const
    {
        return mainAxis.d;
    }

    Line3<T> mainAxis;
    T radius = 0;
    T length = 0;
};

} // namespace MR
