#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

struct MeshLoadSettings
{
    VertColors* colors = nullptr;
    int* deletedFaceCount;
    int* duplicatedVertexCount;
    ProgressCallback callback = {};
};

}
