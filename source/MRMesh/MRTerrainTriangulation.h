#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

// Creates Delaunay triangulation using only XY components of points 
// points will be changed inside this function take argument by value
[[nodiscard]] MRMESH_API Expected<Mesh> terrainTriangulation( std::vector<Vector3f> points, ProgressCallback cb = {} );

}