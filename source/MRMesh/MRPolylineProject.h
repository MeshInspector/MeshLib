#pragma once
#include "MRId.h"
#include "MRVector3.h"
#include <cfloat>

namespace MR
{

// return closest point on ab segment, use closestPointOnLineSegm( const Vector3<T>& pt, const LineSegm3<T> & l ) instead
[[deprecated]] MRMESH_API Vector3f closestPointOnLine( const Vector3f& pt, const Vector3f& a, const Vector3f& b );

struct PolylineProjectionResult
{
    // closest line id on polyline
    UndirectedEdgeId line;
    // closest point on polyline, transformed by xf if it is given
    Vector3f point;
    // squared distance from pt to proj
    float distSq = 0;
};

// computes the closest point on polyline to given point
MRMESH_API PolylineProjectionResult findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline,
    float upDistLimitSq = FLT_MAX,  //< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
    AffineXf3f* xf = nullptr );   //< polyline-to-point transformation, if not specified then identity transformation is assumed

struct PolylineProjectionWithOffsetResult
{
    // closest line id on polyline
    UndirectedEdgeId line;
    // closest point on polyline, transformed by xf if it is given
    Vector3f point;
    // distance from offset point to proj
    float dist = 0;
};

// computes the closest point on polyline to given point, respecting each edge offset
MRMESH_API PolylineProjectionWithOffsetResult findProjectionOnPolylineWithOffset( const Vector3f& pt, const Polyline3& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, //< offset for each edge of polyline
    float upDistLimit = FLT_MAX,  //< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
    AffineXf3f* xf = nullptr );   //< polyline-to-point transformation, if not specified then identity transformation is assumed

// computes the closest point on the mesh edges (specified by the tree) to given point
MRMESH_API PolylineProjectionResult findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree,
    float upDistLimitSq = FLT_MAX,  //< upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
    AffineXf3f* xf = nullptr );   //< polyline-to-point transformation, if not specified then identity transformation is assumed

} //namespace MR
