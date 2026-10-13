#pragma once

#include "MRVector3.h"
#include "MRTriPoint.h"
#include <cstdint>
#include <limits>

namespace MR
{

template<class T>
struct TriTriDistanceResult
{
    /// If the triangles are disjoint, these points are the closest points of
    /// the corresponding triangles. However, if the triangles overlap, these
    /// are basically a random pair of points from the triangles, not
    /// coincident points on the intersection of the triangles, as might
    /// be expected.
    Vector3<T> a, b;

    /// If the triangles are disjoint, it is the squared distance them
    /// (equal to the squared distance between a and b points).
    /// If the triangles overlap, it is zero.
    T distSq = 0;

    /// overlap==true means that the triangles are intersecting and not just touch one another => distSq == 0
    /// if distSq == 0 and overlap == false, then the triangles just touch one another
    bool overlap = true;

    /// the edges of the triangles with points a and b, numbered as in TriPoint::onEdge() (which also contain a point found in a vertex),
    /// or -1 if the point is inside its triangle
    std::int8_t aEdge = -1, bEdge = -1;

    /// barycentric coordinates of point a in triangle ta, given to findTriTriDistance as the first one;
    /// unlike the ones computed from the position of the point, they place a point found on an edge or in a vertex there
    [[nodiscard]] TriPoint<T> aBary( const Triangle3<T>& ta ) const { return TriPoint<T>::fromPoint( a, ta[0], ta[1], ta[2], aEdge ); }

    /// barycentric coordinates of point b in triangle tb, given to findTriTriDistance as the second one
    [[nodiscard]] TriPoint<T> bBary( const Triangle3<T>& tb ) const { return TriPoint<T>::fromPoint( b, tb[0], tb[1], tb[2], bEdge ); }
};
using TriTriDistanceResultf = TriTriDistanceResult<float>;
using TriTriDistanceResultd = TriTriDistanceResult<double>;

enum class UpLimitCheck : bool
{
    GreaterOrEqual, ///< findTriTriDistance exits earlier if (distSqLowerBound >= upDistLimitSq)
    Greater         ///< findTriTriDistance exits earlier if (distSqLowerBound >  upDistLimitSq)
};

template<class T>
struct TriTriDistanceParams
{
    /// upper limit on the distance in question, if the real distance is larger (or equal depending on upLimitCheck)
    /// then findTriTriDistance exits earlier returning lower bound on distSq >= upDistLimitSq
    /// and the points a and b with their edges can be arbitrary
    T upDistLimitSq = std::numeric_limits<T>::max();

    UpLimitCheck upLimitCheck = UpLimitCheck::Greater;

    bool canExitEarlier() const // with the current parameters
    {
        return upDistLimitSq < std::numeric_limits<T>::max();
    }

    bool canExitEarlier( T distSqLowerBound ) const // with this particular lower bound
    {
        return distSqLowerBound > upDistLimitSq || ( upLimitCheck == UpLimitCheck::GreaterOrEqual && distSqLowerBound == upDistLimitSq );
    }
};
using TriTriDistanceParamsf = TriTriDistanceParams<float>;
using TriTriDistanceParamsd = TriTriDistanceParams<double>;

/// computes the closest points on two triangles
[[nodiscard]] MRMESH_API TriTriDistanceResultf findTriTriDistance( const Triangle3f& a, const Triangle3f& b, const TriTriDistanceParamsf& params = {} );
[[nodiscard]] MRMESH_API TriTriDistanceResultd findTriTriDistance( const Triangle3d& a, const Triangle3d& b, const TriTriDistanceParamsd& params = {} );

// This version is not in the bindings, because the pointer parameters are assumed to point to single objects, which is wrong here.
[[deprecated( "Use findDistance() instead" )]] MRMESH_API MR_BIND_IGNORE float triDist( Vector3f & p, Vector3f & q, const Vector3f s[3], const Vector3f t[3] );

[[deprecated( "Use findDistance() instead" )]] MRMESH_API float triDist( Vector3f & p, Vector3f & q, const std::array<Vector3f, 3> & s, const std::array<Vector3f, 3> & t );

[[deprecated( "Use findTwoLineSegmClosestPoints() instead" )]] MRMESH_API void segPoints(
          Vector3f & VEC,
          Vector3f & X, Vector3f & Y,             // closest points
          const Vector3f & P, const Vector3f & A, // seg 1 origin, vector
          const Vector3f & Q, const Vector3f & B);// seg 2 origin, vector

} // namespace MR
