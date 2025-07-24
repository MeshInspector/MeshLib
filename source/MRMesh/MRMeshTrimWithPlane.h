#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRPlane3.h"
#include <functional>
namespace MR
{

/// subdivides all triangles intersected by given plane, leaving smaller triangles that only touch the plane;
/// \return all triangles on the positive side of the plane
/// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
/// \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
/// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
MRMESH_API FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void(EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr );

/** \brief trim mesh by plane
  *
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutEdges optionally return newly appeared hole boundary edges
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
  * \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
  */
[[deprecated]] MRMESH_API MR_BIND_IGNORE void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    UndirectedEdgeBitSet * outCutEdges = nullptr, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr );

/** \brief trim mesh by plane
  *
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutContours optionally return newly appeared hole contours where each edge does not have right face
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
  * \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
  */
[[deprecated]] MRMESH_API MR_BIND_IGNORE void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr );

// stores basic params for trimWithPlane function
struct TrimWithPlaneParams
{
    //Input plane to cut mesh with
    Plane3f plane;
    // if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
    float eps = 0;
    // is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
    std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback;
};

// stores optional output params for trimWithPlane function
struct TrimOptionalOutput
{
    // newly appeared hole boundary edges
    UndirectedEdgeBitSet* outCutEdges = nullptr;
    // newly appeared hole contours where each edge does not have right face
    std::vector<EdgeLoop>* outCutContours = nullptr;
    // mapping from newly appeared triangle to its original triangle (part to full)
    FaceHashMap* new2Old = nullptr;
    // left part of the trimmed mesh
    Mesh* otherPart = nullptr;
    // mapping from newly appeared triangle to its original triangle (part to full) in otherPart
    FaceHashMap* otherNew2Old = nullptr;
    // newly appeared hole contours where each edge does not have right face in otherPart
    std::vector<EdgeLoop>* otherOutCutContours = nullptr;
};

/** \brief trim mesh by plane
  *
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param params stores basic params for trimWithPlane function
  * \param optOut stores optional output params for trimWithPlane function
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const TrimWithPlaneParams& params, const TrimOptionalOutput& optOut = {} );

} //namespace MR
