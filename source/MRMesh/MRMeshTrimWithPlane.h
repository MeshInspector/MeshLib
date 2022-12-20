#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// subdivides all triangles intersected by given plane, leaving smaller triangles that only touch the plane;
/// \return all triangles on the positive side of the plane
MRMESH_API FaceBitSet subdivideWithPlane( Mesh & mesh, const Plane3f & plane );

/** \brief trim mesh by plane
  * 
  * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
  * \param mesh Input mesh that will be cut
  * \param plane Input plane to cut mesh with
  * \param outCutContours optionally return newly appeared hole contours where each edge does not have right face
  */
MRMESH_API void trimWithPlane( Mesh& mesh, const Plane3f & plane,
    std::vector<EdgeLoop> * outCutContours = nullptr );

} //namespace MR
