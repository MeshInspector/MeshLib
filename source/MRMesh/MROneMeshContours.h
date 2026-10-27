#pragma once

#include "MRPrecisePredicates3.h"
#include "MRIntersectionContour.h"
#include "MRExpected.h"
#include "MREnums.h"
#include <variant>

namespace MR
{

// Special data to sort intersections more accurate
struct SortIntersectionsData
{
    const Mesh& otherMesh;
    const ContinuousContours& contours;
    ConvertToIntVector converter;
    const AffineXf3f* rigidB2A{nullptr};
    size_t meshAVertsNum;
    bool isOtherA{false};
};

// Simple point on mesh, represented by primitive id and coordinate in mesh space
struct OneMeshIntersection
{
    std::variant<FaceId, EdgeId, VertId> primitiveId;

    Vector3f coordinate;
};

// One contour on mesh
struct OneMeshContour
{
    MiVector<OneMeshIntersection> intersections;
    bool closed{false};
};
/** \ingroup BooleanGroup
  * \brief Special data type for MR::cutMesh
  * 
  * This is special data for MR::cutMesh, you can convert some other contours representation to it by:\n
  * \ref MR::surfaceLineToClosedMeshContour
  * \ref MR::surfacePathsToMeshContours
  */
using OneMeshContours = MiVector<OneMeshContour>;

// Divides faces that fully own contours into 3 parts with center in center mass of one of the face contours
// if there is more than one contour on face it guarantee to subdivide at least one lone contour on this face
// the AABB tree of the mesh (if any) is updated instead of being invalidated
[[deprecated( "cutMesh connects lone contours itself, and the boolean no longer uses this" )]]
MRMESH_API void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceHashMap* new2oldMap = nullptr );

/// Converts contours given in topological terms as the intersections of one mesh's edge and another mesh's triangle (ContinuousContours),
/// into contours of meshA and/or meshB given as a sequence of (primitiveId and Cartesian coordinates);
/// converters are required for better precision in case of degenerations;
/// note that contours should not have intersections
MRMESH_API void getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours,
    OneMeshContours* outA, OneMeshContours* outB,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr,
    Contours3f* outPtsA = nullptr,
    bool addSelfyTerminalVerts = false ); ///< if true, then open self-intersection contours will be prolonged to terminal vertices

// Converts ordered continuous self contours of single meshes to OneMeshContours
// converters are required for better precision in case of degenerations
[[nodiscard]]
MRMESH_API OneMeshContours getOneMeshSelfIntersectionContours( const Mesh& mesh, const ContinuousContours& contours,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr );

// Converts OneMeshContours contours representation to Contours3f: set of coordinates
[[nodiscard]]
MRMESH_API Contours3f extractMeshContours( const OneMeshContours& meshContours );

using MeshTriPointsConnector = std::function<Expected<SurfacePath>( const MeshTriPoint& start, const MeshTriPoint& end, int startIndex, int endIndex )>;
/** \ingroup BooleanGroup
  * \brief Makes continuous contour by mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds paths between neighbor \p surfaceLine with MeshTriPointsConnector function and build contour MR::cutMesh input
  * \param connectorFn function to build path between neighbor surfaceLine, if not present simple geodesic path function is used
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  */
[[nodiscard]]
MR_BIND_IGNORE MRMESH_API Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    MeshTriPointsConnector connectorFn, std::vector<int>* pivotIndices = nullptr );

/// Geo path search settings
struct SearchPathSettings
{
    GeodesicPathApprox geodesicPathApprox{ GeodesicPathApprox::DijkstraAStar }; ///< the algorithm to compute approximately geodesic path
    int maxReduceIters{ 100 }; ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
};

/** \ingroup BooleanGroup
  * \brief Makes continuous contour by mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds shortest paths between neighbor \p surfaceLine and build contour MR::cutMesh input
  * \param searchSettings settings for search geo path 
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr );

/** \ingroup BooleanGroup
  * \brief Makes closed continuous contour by mesh tri points, note that first and last meshTriPoint should not be same
  * 
  * Finds shortest paths between neighbor \p surfaceLine and build closed contour MR::cutMesh input
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  * \note better use surfaceLineToMeshContour(...) instead, note that it requires same front and back MeshTriPoints for closed contour
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContour> surfaceLineToClosedMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePath to OneMeshContours
  *
  * Creates MR::OneMeshContour object from given surface path with ends for MR::cutMesh input
  * `start` and surfacePath.front() should be from same face
  * surfacePath.back() and `end` should be from same face
  * 
  * note that whole path (including `start` and `end`) should not have self-intersections
  * also following case is not supported (vertex -> edge (incident with vertex)):
  * 
  * vert path  edge point path     edge end
  * o----------o-  --  --  --  --  O
  *  \          \                /
  *       \      \          /
  *            \  \     /
  *               \\/
  *                 o path
  */
[[nodiscard]]
MRMESH_API OneMeshContour surfacePathWithEndsToMeshContour( const Mesh& mesh, 
                                                                          const MeshTriPoint& start, 
                                                                          const SurfacePath& surfacePath, 
                                                                          const MeshTriPoint& end );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePaths to OneMeshContours
  * 
  * Creates MR::OneMeshContours object from given surface paths for MR::cutMesh input
  */
[[nodiscard]]
MRMESH_API OneMeshContours surfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths );

// the old names stay in the bindings, since scripts call them
[[deprecated( "use surfaceLineToMeshContour" )]] [[nodiscard]]
MR_BIND_IGNORE inline Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    MeshTriPointsConnector connectorFn, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToMeshContour( mesh, surfaceLine, std::move( connectorFn ), pivotIndices );
}

[[deprecated( "use surfaceLineToMeshContour" )]] [[nodiscard]]
inline Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToMeshContour( mesh, surfaceLine, searchSettings, pivotIndices );
}

[[deprecated( "use surfaceLineToClosedMeshContour" )]] [[nodiscard]]
inline Expected<OneMeshContour> convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr )
{
    return surfaceLineToClosedMeshContour( mesh, surfaceLine, searchSettings, pivotIndices );
}

[[deprecated( "use surfacePathWithEndsToMeshContour" )]] [[nodiscard]]
inline OneMeshContour convertSurfacePathWithEndsToMeshContour( const Mesh& mesh, const MeshTriPoint& start, const SurfacePath& surfacePath, const MeshTriPoint& end )
{
    return surfacePathWithEndsToMeshContour( mesh, start, surfacePath, end );
}

[[deprecated( "use surfacePathsToMeshContours" )]] [[nodiscard]]
inline OneMeshContours convertSurfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths )
{
    return surfacePathsToMeshContours( mesh, surfacePaths );
}
} //namespace MR
