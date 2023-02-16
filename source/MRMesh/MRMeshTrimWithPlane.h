#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// subdivides all triangles intersected by given plane, leaving smaller triangles that only touch the plane;
/// \return all triangles on the positive side of the plane
/// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
/// \param posEps if existing vertex on positive side of the plane is within posEps distance, then move the vertex not introducing new ones
MRMESH_API FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane, FaceHashMap * new2Old = nullptr, float posEps = 0 );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutEdges optionally return newly appeared hole boundary edges
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param posEps if existing vertex on positive side of the plane is within posEps distance, then move the vertex not introducing new ones
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    UndirectedEdgeBitSet * outCutEdges = nullptr, FaceHashMap * new2Old = nullptr, float posEps = 0 );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutContours optionally return newly appeared hole contours where each edge does not have right face
  * \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
  * \param posEps if existing vertex on positive side of the plane is within posEps distance, then move the vertex not introducing new ones
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    std::vector<EdgeLoop> * outCutContours, FaceHashMap * new2Old = nullptr, float posEps = 0 );

} //namespace MR
