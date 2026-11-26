#pragma once
#include <MRMesh/MRMeshFwd.h>
#include "exports.h"

namespace MR
{

// For an object, find the average direction of its surface
// The returned value is in local coordinates, normalized
// Supports meshes and point clouds (with or without normals), does not process child objects
// On failure, returns a zero vector (though can return zero for valid objects, e.g. cube)
MRVIEWER_API Vector3f findAverageSurfaceVector( const VisualObject &obj );

}
