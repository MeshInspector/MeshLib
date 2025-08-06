#pragma once

#include "MRBitSet.h"
#include "MRAffineXf3.h"
#include "MROneMeshContours.h"

namespace MR
{

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
    FaceBitSet fbsWithContourIntersections;
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


/// Cuts \p mesh by \p contour by projecting all the points
/// \param xf transformation from the CSYS of \p contour to the CSYS of \p mesh
/// \note \p mesh is modified, see \ref cutMesh for info
/// \return Faces to the left of the polyline
[[nodiscard]] MRMESH_API Expected<FaceBitSet> cutMeshByContour( Mesh& mesh, const Contour3f& contour, const AffineXf3f& xf = {} );

/// Cuts \p mesh by \p contours by projecting all the points
/// \param xf transformation from the CSYS of \p contour to the CSYS of \p mesh
/// \note \p mesh is modified, see \ref cutMesh for info
/// \return Faces to the left of the polyline
[[nodiscard]] MRMESH_API Expected<FaceBitSet> cutMeshByContours( Mesh& mesh, const Contours3f& contours, const AffineXf3f& xf = {} );

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

} //namespace MR
