#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRBitSet.h"
#include "MREIntersectionContour.h"
#include "MRMesh/MRExtractIsolines.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include <variant>

namespace MRE
{

// Special data to sort intersections more accurate
struct SortIntersectionsData
{
    const MR::Mesh& otherMesh;
    const ContinuousContours& contours;
    MR::ConvertToIntVector converter;
    const MR::AffineXf3f* rigidB2A{nullptr};
    size_t meshAVertsNum;
    bool isOtherA{false};
};

// Simple point on mesh, represented by primitive id and coordinate in mesh space
struct OneMeshIntersection
{
    enum VariantIndex { Face, Edge, Vertex };
    std::variant<MR::FaceId, MR::EdgeId, MR::VertId> primitiveId;

    MR::Vector3f coordinate;
};

// One contour on mesh
struct OneMeshContour
{
    std::vector<OneMeshIntersection> intersections;
    bool closed{false};
};
/** \ingroup BooleanGroup
  * \brief Special data type for MRE::cutMesh
  * 
  * This is special data for MRE::cutMesh, you can convert some other contours representation to it by:\n
  * \ref MRE::convertMeshTriPointsToClosedContour
  * \ref MRE::convertSurfacePathsToMeshContours
  */
using OneMeshContours = std::vector<OneMeshContour>;

// Divides faces that fully own contours int 3 parts with center in contours center mass
MREALGORITHMS_API void subdivideLoneContours( MR::Mesh& mesh, const OneMeshContours& contours, MR::FaceMap* new2oldMap = nullptr );

// Converts ordered continuous contours of two meshes to OneMeshContours
// converters is required for better precision in case of degenerations
// note that contours should not have intersections
MREALGORITHMS_API OneMeshContours getOneMeshIntersectionContours( const MR::Mesh& meshA, const MR::Mesh& meshB, const ContinuousContours& contours, bool getMeshAIntersections,
    const MR::CoordinateConverters& converters, const MR::AffineXf3f* rigidB2A = nullptr );

/** \ingroup BooleanGroup
  * \brief Makes closed continuous contour by mesh tri points
  * 
  * Finds shortest paths between neighbor \p meshTriPoints and build closed contour MRE::cutMesh input
  */
MREALGORITHMS_API OneMeshContour convertMeshTriPointsToClosedContour( const MR::Mesh& mesh, const std::vector<MR::MeshTriPoint>& meshTriPoints );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePath to OneMeshContours
  *
  * Creates MRE::OneMeshContour object from given surface path with ends for MRE::cutMesh input
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
MREALGORITHMS_API OneMeshContour convertSurfacePathWithEndsToMeshContour( const MR::Mesh& mesh, 
                                                                          const MR::MeshTriPoint& start, 
                                                                          const MR::SurfacePath& surfacePath, 
                                                                          const MR::MeshTriPoint& end );

/** \ingroup BooleanGroup
  * \brief Converts SurfacePaths to OneMeshContours
  * 
  * Creates MRE::OneMeshContours object from given surface paths for MRE::cutMesh input
  */
MREALGORITHMS_API OneMeshContours convertSurfacePathsToMeshContours( const MR::Mesh& mesh, const std::vector<MR::SurfacePath>& surfacePaths );

/** \struct MRE::CutMeshParameters
  * \ingroup BooleanGroup
  * \brief Parameters of MRE::cutMesh
  * 
  * This structure contains some options and optional outputs of MRE::cutMesh function
  * \sa \ref MRE::CutMeshResult
  */
struct CutMeshParameters
{
    /// This is optional input for better contours resolving\n
    /// it provides additional info from other mesh used in boolean operation, useful to solve some degeneration
    /// \note Most likely you don't need this in case you call MRE::cutMesh manualy, use case of it is MRE::boolean
    const SortIntersectionsData* sortData{nullptr};
    /// This is optional output - map from newly generated faces to old faces (N-1)
    MR::FaceMap* new2OldMap{nullptr};
    /// If this flag is set, MRE::cutMesh will fill all possible triangles, except bad ones; otherwise it will leave deleted faces on all contours line (only in case of bad triangles)
    /// \note Bad triangles here mean faces where contours have intersections and cannot be cut and filled in an good way
    bool forceFillAfterBadCut{false};
};

/** \struct MRE::CutMeshResult
  * \ingroup BooleanGroup
  * This structure contains result of MRE::cutMesh function
  */
struct CutMeshResult
{
    /// Paths of new edges on mesh, they represent same contours as input, but already cut
    std::vector<MR::EdgePath> resultCut;
    /// Bitset of bad triangles - triangles where input contours have intersections and cannot be cut and filled in a good way
    /// \sa \ref MRE::CutMeshParameters
    MR::FaceBitSet fbsWithCountourIntersections;
};

/** \ingroup BooleanGroup
  * \brief Cuts mesh by given contours
  * 
  * This function cuts mesh making new edges paths on place of input contours
  * \param mesh Input mesh that will be cut
  * \param contours Input contours to cut mesh with, find more \ref MRE::OneMeshContours
  * \param params Parameters describing some cut options, find more \ref MRE::CutMeshParameters
  * \return New edges that correspond to given contours, find more \ref MRE::CutMeshResult
  * \parblock
  * \warning Input contours should have no intersections, faces where contours intersects (`bad faces`) will not be allowed for fill
  * \endparblock
  * \parblock
  * \warning Input mesh will be changed in any case, if `bad faces` are in mesh, mesh will be spoiled, \n
  * so if you cannot guarantee contours without intersections better make copy of mesh, before using this function
  * \endparblock
  */
MREALGORITHMS_API CutMeshResult cutMesh( MR::Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params = {} );

/** \ingroup BooleanGroup
  * \brief Simple cut mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param mapNew2Old (this is optional output) map from newly generated faces to old faces (N-1)
  * \note This function changes input mesh
  * \return New edges that correspond to given contours, find more \ref MRE::CutMeshResult
  */
MREALGORITHMS_API std::vector<MR::EdgePath> cutMeshWithPlane( MR::Mesh& mesh, const MR::Plane3f& plane, MR::FaceMap* mapNew2Old = nullptr );

}
