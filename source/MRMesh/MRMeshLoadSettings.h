#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

// structure with settings and side output parameters for loading mesh
struct MeshLoadSettings
{
    VertColors* colors = nullptr;    ///< points where to load vertex color map
    VertNormals* normals = nullptr;  ///< points where to load vertex normals
    int* skippedFaceCount = nullptr; ///< counter of skipped faces (faces than can't be created)
    int* duplicatedVertexCount = nullptr; ///< counter of duplicated vertices (that created for resolve non-manifold geometry)
    AffineXf3f* xf = nullptr; ///< transform for the loaded mesh
    ProgressCallback callback;       ///< callback for set progress and stop process
};

}
