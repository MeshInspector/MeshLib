#pragma once
#pragma once
#include "MRMeshFwd.h"

namespace MR
{

MRMESH_API std::vector<std::pair<EdgePoint, EdgePoint>> trimPolylineWithPlane( const Polyline3& polyline, const Plane3f& plane, float eps );

}