#pragma once
#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

namespace PlanarTriangulation
{

using HoleVertIds = std::vector<VertId>;
using HolesVertIds = std::vector<HoleVertIds>;

/// return vertices of holes that correspond internal contours representation of PlanarTriangulation
MRMESH_API HolesVertIds findHoleVertIdsByHoleEdges( const MeshTopology& tp, const std::vector<EdgePath>& holePaths );

/**
 * @brief triangulate 2d contours
 * @detail only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
 * @return return created mesh
 */
MRMESH_API Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds = nullptr );
MRMESH_API Mesh triangulateContours( const Contours2f& contours, const HolesVertIds* holeVertsIds = nullptr );

/**
 * @brief triangulate 2d contours
 * @detail only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
 * @return std::optional<Mesh> : if some contours intersect return false, otherwise return created mesh
 */
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds = nullptr );
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds = nullptr );

}
}