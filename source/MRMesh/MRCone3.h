#pragma once
#include "MRLine3.h"

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

    Vector3<T> projectPoint( const Vector3<T>& point ) const
    {
        // Get direction, center, and angle of the cone from the specified viewport
        const Vector3<T>& n = direction();
        const Vector3<T>& center = apex();
        const T coneAngle = angle;

        // Calculate vector X as the difference between the point and the center of the cone
        auto X = point - center;

        // Calculate the angle between vectors n (cone main axis)  and X (center normalyzed source point)
        T angleX = std::atan2( cross( n, X ).length(), dot( n, X ) );

        // This means the projection will fall on the apex of the cone.
        if ( coneAngle + PI_F / 2.0 < angleX )
            return  center;

        // Project vector X onto the cone main axis
        auto K = n * MR::dot( X, n );
        auto XK = ( X - K );

        // We find the point of intersection of the vector XK with the surface of the cone 
        // and find a guide ventor along the surface of the cone to the projection point
        auto D = K + XK.normalized() * ( K.length() * std::tan( coneAngle ) );
        auto normD = D.normalized();
        // Calculate the projected point on the conical surface
        return normD * dot( normD, X ) + center;
    }


    Line3<T> axis; // the combination of the apex of the cone and the direction of its main axis in space. 
    // for any cone point dot( point , direction ) >=0
    T angle = 0;       // cone angle, main axis vs side
    T height = 0;      // cone height
};



} // namespace MR
