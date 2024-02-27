#pragma once
#pragma once
#include "MRMeshFwd.h"

namespace MR
{
/// This function splits edges intersected by the plane
/// \return New edges with origin on the plane and oriented to the positive direction
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
MRMESH_API EdgeBitSet subdividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane );

/// This function divides polyline with a plane
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param otherPart Optional return, polyline composed from edges on the negative side of the plane
MRMESH_API void dividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, Polyline3* otherPart = nullptr );

/// This function cuts polyline with a plane
/// \return Edge segments tha are closer to the plane than \param eps
/// \param polyline Input polyline that will be cut by the plane
/// \param plane Input plane to cut polyline with
/// \param eps Maximal distance from the plane
MRMESH_API std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps );

}