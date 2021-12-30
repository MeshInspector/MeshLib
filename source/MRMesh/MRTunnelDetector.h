#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh
MRMESH_API std::vector<EdgeLoop> detectBasisTunnels( const MeshPart & mp );

// returns tunnels as a number of faces;
// if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere
MRMESH_API FaceBitSet detectTunnelFaces( const MeshPart & mp, float maxTunnelLength );

} //namespace MR
