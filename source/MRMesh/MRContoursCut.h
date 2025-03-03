#pragma once

#include "MRVector3.h"
#include "MRId.h"
#include "MRBitSet.h"
#include "MRIntersectionContour.h"
#include "MRExtractIsolines.h"
#include "MRMeshCollidePrecise.h"
#include "MRSurfacePath.h"
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
    enum VariantIndex { Face, Edge, Vertex };
    std::variant<FaceId, EdgeId, VertId> primitiveId;

    Vector3f coordinate;
};

// One contour on mesh
struct OneMeshContour
{
    std::vector<OneMeshIntersection> intersections;
    bool closed{false};
};
/** \ingroup BooleanGroup
  * \brief Special data type for MR::cutMesh
  * 
  * This is special data for MR::cutMesh, you can convert some other contours representation to it by:\n
  * \ref MR::convertMeshTriPointsToClosedContour
  * \ref MR::convertSurfacePathsToMeshContours
  */
using OneMeshContours = std::vector<OneMeshContour>;

// Divides faces that fully own contours into 3 parts with center in center mass of one of the face contours
// if there is more than one contour on face it guarantee to subdivide at least one lone contour on this face
MRMESH_API void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceHashMap* new2oldMap = nullptr );

// Converts ordered continuous contours of two meshes to OneMeshContours
// converters is required for better precision in case of degenerations
// note that contours should not have intersections
[[nodiscard]]
MRMESH_API OneMeshContours getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours, bool getMeshAIntersections,
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
MR_BIND_IGNORE MRMESH_API Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
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
MRMESH_API Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    SearchPathSettings searchSettings = {}, std::vector<int>* pivotIndices = nullptr );

/** \ingroup BooleanGroup
  * \brief Makes continuous contour by iso-line from mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds shortest paths between neighbor \p surfaceLine and build offset contour on surface for MR::cutMesh input
  * \param offset amount of offset form given point, note that absolute value is used and isoline in both direction returned
  * \param searchSettings settings for search geodesic path
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContours> convertMeshTriPointsSurfaceOffsetToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    float offset, SearchPathSettings searchSettings = {} );

/** \ingroup BooleanGroup
  * \brief Makes continuous contour by iso-line from mesh tri points, if first and last meshTriPoint is the same, makes closed contour
  *
  * Finds shortest paths between neighbor \p surfaceLine and build offset contour on surface for MR::cutMesh input
  * \param offsetAtPoint functor that returns amount of offset form arg point, note that absolute value is used and isoline in both direction returned
  * \param searchSettings settings for search geodesic path
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContours> convertMeshTriPointsSurfaceOffsetToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
    const std::function<float(int)>& offsetAtPoint, SearchPathSettings searchSettings = {});

/** \ingroup BooleanGroup
  * \brief Makes closed continuous contour by mesh tri points, note that first and last meshTriPoint should not be same
  * 
  * Finds shortest paths between neighbor \p surfaceLine and build closed contour MR::cutMesh input
  * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
  * \note better use convertMeshTriPointsToMeshContour(...) instead, note that it requires same front and back MeshTriPoints for closed contour
  */
[[nodiscard]]
MRMESH_API Expected<OneMeshContour> convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine,
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
MRMESH_API OneMeshContour convertSurfacePathWithEndsToMeshContour( const Mesh& mesh, 
                                                                          const MeshTriPoint& start, 
                                                                          const SurfacePath& surfacePath, 
                                                                          const MeshTriPoint& end );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePaths to OneMeshContours
  * 
  * Creates MR::OneMeshContours object from given surface paths for MR::cutMesh input
  */
[[nodiscard]]
MRMESH_API OneMeshContours convertSurfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths );

/// \ingroup BooleanGroup
/// Map structure to find primitives of old topology by edges introduced in cutMesh
struct NewEdgesMap
{
    /// true here means that a subdivided edge is a part of some original edge edge before mesh subdivision;
    /// false here is both for unmodified edges and for new edges introduced within original triangles
    UndirectedEdgeBitSet splitEdges;

    /// maps every edge appeared during subdivision to an original edge before mesh subdivision;
    /// for splitEdges[key]=true, the value is arbitrary oriented original edge, for which key-edge is its part;
    /// for splitEdges[key]=false, the value is an original triangle
    HashMap<UndirectedEdgeId, int> map;
};

/** \struct MR::CutMeshParameters
  * \ingroup BooleanGroup
  * \brief Parameters of MR::cutMesh
  * 
  * This structure contains some options and optional outputs of MR::cutMesh function
  * \sa \ref MR::CutMeshResult
  */
struct CutMeshParameters
{
    /// This is optional input for better contours resolving\n
    /// it provides additional info from other mesh used in boolean operation, useful to solve some degeneration
    /// \note Most likely you don't need this in case you call MR::cutMesh manualy, use case of it is MR::boolean
    const SortIntersectionsData* sortData{nullptr};
    /// This is optional output - map from newly generated faces to old faces (N-1)
    FaceMap* new2OldMap{nullptr};
    /// This enum defines the MR::cutMesh behaviour in case of bad faces acure
    /// basicaly MR::cutMesh removes all faces which contours pass through, adds new edges to topology and fills all removed parts
    /// 
    /// \note Bad faces here mean faces where contours have intersections and cannot be cut and filled in an good way
    enum class ForceFill
    {
        None, //< if bad faces occur does not fill anything
        Good, //< fills all faces except bad ones
        All   //< fills all faces with bad ones, but on bad faces triangulation can also be bad (may have self-intersections or tunnels)
    } forceFillMode{ ForceFill::None };

    /// Optional output map for each new edge introduced after cut maps edge from old topology or old face
    NewEdgesMap* new2oldEdgesMap{ nullptr };
};

/** \struct MR::CutMeshResult
  * \ingroup BooleanGroup
  * This structure contains result of MR::cutMesh function
  */
struct CutMeshResult
{
    /// Paths of new edges on mesh, they represent same contours as input, but already cut
    std::vector<EdgePath> resultCut;
    /// Bitset of bad triangles - triangles where input contours have intersections and cannot be cut and filled in a good way
    /// \sa \ref MR::CutMeshParameters
    FaceBitSet fbsWithCountourIntersections;
};

/** \ingroup BooleanGroup
  * \brief Cuts mesh by given contours
  * 
  * This function cuts mesh making new edges paths on place of input contours
  * \param mesh Input mesh that will be cut
  * \param contours Input contours to cut mesh with, find more \ref MR::OneMeshContours
  * \param params Parameters describing some cut options, find more \ref MR::CutMeshParameters
  * \return New edges that correspond to given contours, find more \ref MR::CutMeshResult
  * \parblock
  * \warning Input contours should have no intersections, faces where contours intersects (`bad faces`) will not be allowed for fill
  * \endparblock
  * \parblock
  * \warning Input mesh will be changed in any case, if `bad faces` are in mesh, mesh will be spoiled, \n
  * so if you cannot guarantee contours without intersections better make copy of mesh, before using this function
  * \endparblock
  */
MRMESH_API CutMeshResult cutMesh( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params = {} );


/// Cuts \p mesh by \p polyline by projecting all the points
/// \note \p mesh is modified, see \ref cutMesh for info
/// \return Faces to the left of the polyline
MRMESH_API Expected<FaceBitSet> cutMeshByPolyline( Mesh& mesh, const AffineXf3f& meshXf,
                                                   const Polyline3& polyline, const AffineXf3f& lineXf );

} //namespace MR
