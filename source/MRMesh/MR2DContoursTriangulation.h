#pragma once
#include "MRMeshFwd.h"


namespace MR
{

namespace PlanarTriangulation
{
// triangulate 2d contours
// only closed contours are allowed 
// (first point of each contour should be the same as last point of the contour)
MRMESH_API std::optional<Mesh> triangulateContours( const Contours2d& contours, bool abortWhenIntersect = false );
MRMESH_API std::optional<Mesh> triangulateContours( const Contours2f& contours, bool abortWhenIntersect = false );

}
}