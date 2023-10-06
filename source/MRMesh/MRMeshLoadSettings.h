#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

// structure with settings and side output parameters for loading mesh
struct MeshLoadSettings
{
    VertColors* colors = nullptr; // vertices color map
    int* deletedFaceCount = nullptr; // counter of deleted faces (faces than can't be created)
    int* duplicatedVertexCount = nullptr; // counter of duplicated vertices (that created for resolve non-manifold geometry)
    ProgressCallback callback = {}; // callback for set progress and stop process
};

}
