#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct SphereParams
{
    float radius = 1;
    int numMeshVertices = 100;
};

// creates a mesh of sphere with irregular triangulation
MRMESH_API Mesh makeSphere( const SphereParams & params );

} //namespace MR
