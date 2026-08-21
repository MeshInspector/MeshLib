#pragma once

#include "MRMeshFwd.h"
#include "MRPrecisePredicates3.h"
#include "MRFastInt128.h"
#include "MRVector.h"
#include "MRPch/MRBindingMacros.h"
#include <cstddef>
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

/// the amount of work done during the search of alpha-shape triangles;
/// every function below only increases the counters, never resets them,
/// so the statistics of several calls can be accumulated in one object
struct AlphaShapeStats
{
    /// the number of neighbours found within the search radius, before any of the filters below;
    /// the searches around all the points are quadratic in this, so it explains most of the
    /// difference in the time spent per point between one cloud and another
    std::size_t collectedNeis = 0;

    /// the number of pairs of neighbours checked for one of them making the other redundant,
    /// which is quadratic in the neighbours of a point
    std::size_t redundancyTests = 0;

    /// the number of neighbours found redundant by those checks and excluded from the search
    std::size_t redundantNeis = 0;

    /// the number of triangles considered: the triples of close enough points that were checked
    /// for the existence of a ball of the given radius passing via all three of them
    std::size_t consideredTris = 0;

    /// the number of considered triangles touchable by the ball of the given radius,
    /// for each of which two balls (one from each side of the triangle) were tested for emptiness
    std::size_t touchableTris = 0;

    /// the number of point-in-ball tests performed for the balls of touchable triangles;
    /// less than 2 * touchableTris * (points in the neighbourhood) because a ball
    /// is not tested further as soon as the first point inside it is found
    std::size_t inBallTests = 0;

    /// the number of neighbours tested for being shadowed by the balls of a touchable triangle
    std::size_t shadowTests = 0;

    /// the number of shadow tests not decided by the floating-point rejection,
    /// which had to evaluate the exact predicates
    std::size_t exactShadowTests = 0;

    /// the number of neighbours found shadowed, which are excluded from the search
    std::size_t shadowedNeis = 0;

    AlphaShapeStats & operator +=( const AlphaShapeStats & r )
    {
        collectedNeis += r.collectedNeis;
        redundancyTests += r.redundancyTests;
        redundantNeis += r.redundantNeis;
        consideredTris += r.consideredTris;
        touchableTris += r.touchableTris;
        inBallTests += r.inBallTests;
        shadowTests += r.shadowTests;
        exactShadowTests += r.exactShadowTests;
        shadowedNeis += r.shadowedNeis;
        return *this;
    }
};

/// a neighbour of the point #v in the search of alpha-shape triangles around it
struct AlphaShapeNei
{
    /// the neighbour's id together with its integer coordinates
    PreciseVertCoords coords;

    /// the exact squared distance from #v in the units of the same integer grid
    MR_BIND_IGNORE FastInt128 distSq;
};

/// finds all triangles of alpha-shape with negative alpha = -1/radius,
/// where each triangle contains point #v and two other points;
/// the valid points sharing one position in the integer grid are merged: only the point with
/// the smallest id among them appears in the triangles (in particular, #v itself gets no
/// triangles if it has such a twin), so the triangles found around all the points agree
MRMESH_API void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v,
    const AlphaShapeData & data, ///< prepared by getAlphaShapeData for the same cloud and the same radius
    Triangulation & appendTris,  ///< found triangles will be appended here
    std::vector<AlphaShapeNei> & neis, ///< temporary storage to avoid memory allocations, it will be filled with the neighbours of point #v within data.searchRadius sorted by distance, except the duplicates and the ones redundant for the search; cleared if #v duplicates a point with a smaller id
    bool onlyLargerVids,         ///< if true then two other points must have larger ids (to avoid finding same triangles several times)
    AlphaShapeStats * stats = nullptr ); ///< optional statistics of the work done, which is increased here

/// finds all triangles of alpha-shape with negative alpha = -1/radius
[[nodiscard]] MRMESH_API std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, float radius,
    const ProgressCallback & cb, AlphaShapeStats * stats = nullptr );
[[nodiscard]] MRMESH_API Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius,
    AlphaShapeStats * stats = nullptr );

/// finds all triangles of alpha-shape given the data prepared by getAlphaShapeData for the same cloud
/// (preferably with allPoints=true, since the triangles around all points will be searched)
[[nodiscard]] MRMESH_API std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud,
    const AlphaShapeData & data, const ProgressCallback & cb, AlphaShapeStats * stats = nullptr );

/// builds alpha-shape mesh with negative alpha = -1/radius
[[nodiscard]] MRMESH_API std::optional<Mesh> findAlphaShape( const PointCloud & cloud, float radius,
    const ProgressCallback & cb, AlphaShapeStats * stats = nullptr );
[[nodiscard]] MRMESH_API Mesh findAlphaShape( const PointCloud & cloud, float radius, AlphaShapeStats * stats = nullptr );

} //namespace MR
