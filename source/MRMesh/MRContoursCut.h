#pragma once

#include "MRVector3.h"
#include "MRId.h"
#include "MRBitSet.h"
#include "MRIntersectionContour.h"
#include "MRExtractIsolines.h"
#include "MRMeshCollidePrecise.h"
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

// Divides faces that fully own contours int 3 parts with center in contours center mass
MRMESH_API void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceMap* new2oldMap = nullptr );

// Converts ordered continuous contours of two meshes to OneMeshContours
// converters is required for better precision in case of degenerations
// note that contours should not have intersections
MRMESH_API OneMeshContours getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours, bool getMeshAIntersections,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr );

/** \ingroup BooleanGroup
  * \brief Makes closed continuous contour by mesh tri points
  * 
  * Finds shortest paths between neighbor \p meshTriPoints and build closed contour MR::cutMesh input
  */
MRMESH_API OneMeshContour convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPoints );

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
MRMESH_API OneMeshContour convertSurfacePathWithEndsToMeshContour( const Mesh& mesh, 
                                                                          const MeshTriPoint& start, 
                                                                          const SurfacePath& surfacePath, 
                                                                          const MeshTriPoint& end );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePaths to OneMeshContours
  * 
  * Creates MR::OneMeshContours object from given surface paths for MR::cutMesh input
  */
MRMESH_API OneMeshContours convertSurfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths );

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
    /// If this flag is set, MR::cutMesh will fill all possible triangles, except bad ones; otherwise it will leave deleted faces on all contours line (only in case of bad triangles)
    /// \note Bad triangles here mean faces where contours have intersections and cannot be cut and filled in an good way
    bool forceFillAfterBadCut{false};
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

/** \ingroup BooleanGroup
  * \brief Simple cut mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param mapNew2Old (this is optional output) map from newly generated faces to old faces (N-1)
  * \note This function changes input mesh
  * \return New edges that correspond to given contours, find more \ref MR::CutMeshResult
  */
MRMESH_API std::vector<EdgePath> cutMeshWithPlane( Mesh& mesh, const Plane3f& plane, FaceMap* mapNew2Old = nullptr );

} //namespace MR
