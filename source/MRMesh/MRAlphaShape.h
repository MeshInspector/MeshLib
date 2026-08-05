#pragma once

#include "MRMeshFwd.h"
#include "MRPrecisePredicates3.h"
#include "MRVector.h"
#include <optional>

namespace MR
{

// inspired by "On the Shape of a Set of Points in the Plane" by HERBERT EDELSBRUNNER, DAVID G. KIRKPATRICK, AND RAIMUND SEIDEL
// https://www.cs.jhu.edu/~misha/Fall13b/Papers/Edelsbrunner93.pdf

/// a point cloud and a ball radius prepared for the search of alpha-shape triangles;
/// the emptiness of a ball is tested in integer coordinates with simulation-of-simplicity
/// resolving the degeneracies, so both the cloud and the radius are needed in integer form as well
struct AlphaShapeData
{
    /// converts cloud's points in integer coordinates
    ConvertToIntVector toInt;

    /// integer coordinates of all valid points of the cloud,
    /// or empty if they have to be computed on the fly by toInt
    Vector<Vector3i, VertId> intPoints;

    /// squared ball radius in the units of toInt's integer grid, not larger than the squared
    /// original radius (so that no point inside a ball can be farther than searchRadius
    /// from the ball's points), and small enough for the predicates not to overflow
    std::int64_t intRadiusSq = 0;

    /// the radius of the neighbourhood of a point to consider;
    /// a bit larger than the doubled ball radius to compensate the rounding of integer coordinates
    float searchRadius = 0;

    /// returns point #v of the cloud (which must be the cloud given to getAlphaShapeData)
    /// together with its integer coordinates
    [[nodiscard]] MRMESH_API PreciseVertCoords coords( const PointCloud & cloud, VertId v ) const;
};

/// prepares the data for the search of alpha-shape triangles with negative alpha = -1/radius in the cloud
/// \param allPoints whether to convert all valid points of the cloud in integer coordinates in parallel,
///                  which pays off if the triangles around many points will be searched
[[nodiscard]] MRMESH_API AlphaShapeData getAlphaShapeData( const PointCloud & cloud, float radius, bool allPoints );

/// finds all triangles of alpha-shape with negative alpha = -1/radius,
/// where each triangle contains point #v and two other points
MRMESH_API void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v,
    const AlphaShapeData & data, ///< prepared by getAlphaShapeData for the same cloud and the same radius
    Triangulation & appendTris,  ///< found triangles will be appended here
    std::vector<PreciseVertCoords> & neis, ///< temporary storage to avoid memory allocations, it will be filled with all neighbours of point #v within data.searchRadius
    bool onlyLargerVids );       ///< if true then two other points must have larger ids (to avoid finding same triangles several times)

/// finds all triangles of alpha-shape with negative alpha = -1/radius
[[nodiscard]] MRMESH_API std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, float radius, const ProgressCallback & cb );
[[nodiscard]] MRMESH_API Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius );

/// finds all triangles of alpha-shape given the data prepared by getAlphaShapeData for the same cloud
/// (preferably with allPoints=true, since the triangles around all points will be searched)
[[nodiscard]] MRMESH_API std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud,
    const AlphaShapeData & data, const ProgressCallback & cb );

/// builds alpha-shape mesh with negative alpha = -1/radius
[[nodiscard]] MRMESH_API std::optional<Mesh> findAlphaShape( const PointCloud & cloud, float radius, const ProgressCallback & cb );
[[nodiscard]] MRMESH_API Mesh findAlphaShape( const PointCloud & cloud, float radius );

} //namespace MR
