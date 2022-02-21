#pragma once
#include "MRMeshFwd.h"


namespace MR
{

namespace PlanarTriangulation
{
/**
 * @brief triangulate 2d contours
 * @detail only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param mergeClosePoints merge close points in contours
 * @return return created mesh
 */
MRMESH_API Mesh triangulateContours( const Contours2d& contours, bool mergeClosePoints = true );
MRMESH_API Mesh triangulateContours( const Contours2f& contours, bool mergeClosePoints = true );

/**
 * @brief triangulate 2d contours
 * @detail only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param mergeClosePoints merge close points in contours
 * @return std::optional<Mesh> : if some contours intersect return false, otherwise return created mesh
 */
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, bool mergeClosePoints = true );
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, bool mergeClosePoints = true );

}
}