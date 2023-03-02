#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// subdivides all triangles intersected by given plane, leaving smaller triangles that only touch the plane;
/// \return all triangles on the positive side of the plane
/// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
/// \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
/// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives origin vertex, destination vertex, and weight of the origin vertex
MRMESH_API FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void(VertId, VertId, float )> onEdgeSplitCallback = nullptr );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutEdges optionally return newly appeared hole boundary edges
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
  * \param onEdgeSplitCallback is invoked each time when an edge is split. Receives origin vertex, destination vertex, and weight of the origin vertex
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    UndirectedEdgeBitSet * outCutEdges = nullptr, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( VertId, VertId, float )> onEdgeSplitCallback = nullptr );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutContours optionally return newly appeared hole contours where each edge does not have right face
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
  * \param onEdgeSplitCallback is invoked each time when an edge is split. Receives origin vertex, destination vertex, and weight of the origin vertex
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old = nullptr, float eps = 0, std::function<void( VertId, VertId, float )> onEdgeSplitCallback = nullptr );

} //namespace MR
