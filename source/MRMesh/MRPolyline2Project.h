#pragma once

#include "MRId.h"
#include "MRVector2.h"
#include <cfloat>

namespace MR
{

// return closest point on ab segment, use closestPointOnLineSegm( const Vector3<T>& pt, const LineSegm3<T> & l ) instead
[[deprecated]] MRMESH_API Vector2f closestPointOnLine( const Vector2f& pt, const Vector2f& a, const Vector2f& b );

struct Polyline2ProjectionResult
{
    // closest line id on polyline
    UndirectedEdgeId line;
    // closest point on polyline, transformed by xf if it is given
    Vector2f point;
    // squared distance from pt to proj
    float distSq = 0;
};

// computes the closest point on two dimensional polyline to given point
MRMESH_API Polyline2ProjectionResult findProjectionOnPolyline2( const Vector2f& pt, const Polyline2& polyline,
    float upDistLimitSq = FLT_MAX, //< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
    AffineXf2f* xf = nullptr );    //< mesh-to-point transformation, if not specified then identity transformation is assumed

struct Polyline2ProjectionWithOffsetResult
{
    // closest line id on polyline
    UndirectedEdgeId line;
    // closest point on polyline, transformed by xf if it is given
    Vector2f point;
    // distance from offset point to proj
    float dist = 0;
};

// computes the closest point on polyline to given point, respecting each edge offset
MRMESH_API Polyline2ProjectionWithOffsetResult findProjectionOnPolyline2WithOffset( const Vector2f& pt, const Polyline2& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, //< offset for each edge of polyline
    float upDistLimit = FLT_MAX,  //< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
    AffineXf2f* xf = nullptr );   //< polyline-to-point transformation, if not specified then identity transformation is assumed

} //namespace MR
