#pragma once
#include "MRMeshFwd.h"


namespace MR
{

namespace PlanarTriangulation
{
// triangulate 2d contours
// only closed contours are allowed 
// (first point of each contour should be the same as last point of the contour)
MRMESH_API Mesh triangulateContours( const Contours2d& contours );
MRMESH_API Mesh triangulateContours( const Contours2f& contours );

}
}