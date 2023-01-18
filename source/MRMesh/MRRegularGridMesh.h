#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRVector3.h"
#include <functional>

namespace MR
{

// Lambda for validating grid lattice
using RegularGridLatticeValidator = std::function<bool( size_t x, size_t y )>;
// Lambda for getting lattice position
using RegularGridLatticePositioner = std::function<Vector3f( size_t x, size_t y )>;
// Lambda for validating mesh face
using RegularGridMeshFaceValidator =
        std::function<bool( size_t x0, size_t y0, size_t x1, size_t y1, size_t x2, size_t y2 )>;

// Creates regular mesh with points in valid grid lattice
MRMESH_API Mesh makeRegularGridMesh( size_t width, size_t height,
                                          const RegularGridLatticeValidator& validator,
                                          const RegularGridLatticePositioner& positioner,
                                          const RegularGridMeshFaceValidator& faceValidator = {} );
}