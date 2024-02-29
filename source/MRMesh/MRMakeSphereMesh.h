#pragma once

#include "MRMesh.h"

namespace MR
{

struct SphereParams
{
    float radius = 1;
    int numMeshVertices = 100;
};

/// creates a mesh of sphere with irregular triangulation
MRMESH_API Mesh makeSphere( const SphereParams & params );

/// creates a mesh of sphere with regular triangulation (parallels and meridians)
MRMESH_API Mesh makeUVSphere( float radius = 1.0, int horisontalResolution = 16, int verticalResolution = 16 );

} //namespace MR
