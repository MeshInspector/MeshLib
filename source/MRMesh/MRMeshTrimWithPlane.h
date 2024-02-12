#pragma once

#include "MRMeshFwd.h"
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
  * \param otherMesh optionally returns left part of the trimmed mesh
  * \param otherOutCutEdges optionally return newly appeared hole boundary edges in otherMesh
  * \param otherNew2Old receive mapping from newly appeared triangle to its original triangle (part to full) in otherMesh
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    UndirectedEdgeBitSet * outCutEdges = nullptr, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr,
    Mesh* otherMesh = nullptr, UndirectedEdgeBitSet* otherOutCutEdges = nullptr, FaceHashMap* otherNew2Old = nullptr );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutContours optionally return newly appeared hole contours where each edge does not have right face
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
  * \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
  * \param otherMesh optionally returns left part of the trimmed mesh
  * \param otherOutCutContours optionally return newly appeared hole contours in otherMesh where each edge does not have right face
  * \param otherNew2Old receive mapping from newly appeared triangle to its original triangle (part to full) in otherMesh
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback = nullptr,
    Mesh* otherMesh = nullptr, std::vector<EdgeLoop>* otherOutCutContours = nullptr, FaceHashMap* otherNew2Old = nullptr );

} //namespace MR
