#pragma once
#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

namespace PlanarTriangulation
{

/// Specify mode of detecting inside and outside parts of triangulation
enum class WindingMode
{
    NonZero,
    Positive,
    Negative
};

using HoleVertIds = std::vector<VertId>;
using HolesVertIds = std::vector<HoleVertIds>;

/// return vertices of holes that correspond internal contours representation of PlanarTriangulation
MRMESH_API HolesVertIds findHoleVertIdsByHoleEdges( const MeshTopology& tp, const std::vector<EdgePath>& holePaths );

/// struct to map new vertices (only appear on intersections) of the outline to it's edges
struct IntersectionsMap
{
    /// shift of index
    size_t shift{ 0 };
    /// map[id-shift] = {min vert id of lower intersection edge, min vert id of upper intersection edge}
    std::vector<std::pair<int, int>> map;
};

/// returns Mesh with boundaries representing outline if input contours
/// interMap optional output intersection map
MRMESH_API Mesh getOutlineMesh( const Contours2f& contours, IntersectionsMap* interMap = nullptr );

/// maps output contour ids to input ones
struct InputContourId
{
    /// if interId is invalid: id is index in input contours
    /// otherwise minimum id of lower intersection edge
    int id{ -1 };
    /// if valid minimum id of upper intersection edge
    int interId{ -1 }; 
};
using ContourIdMap = std::vector<InputContourId>;
using ContoursIdMap = std::vector<ContourIdMap>;

/// returns Contour representing outline if input contours
/// indicesMap optional output from result contour ids to input ones
MRMESH_API Contours2f getOutline( const Contours2f& contours, ContoursIdMap* indicesMap = nullptr );

/**
 * @brief triangulate 2d contours
 * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
 * @return return created mesh
 */
MRMESH_API Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds = nullptr );
MRMESH_API Mesh triangulateContours( const Contours2f& contours, const HolesVertIds* holeVertsIds = nullptr );

/**
 * @brief triangulate 2d contours
 * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
 * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
 * @return std::optional<Mesh> : if some contours intersect return false, otherwise return created mesh
 */
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds = nullptr );
MRMESH_API std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds = nullptr );

}
}