#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

struct MeshLoadSettings
{
    VertColors* colors = nullptr;
    int* duplicatedEdgeCount;
    int* duplicatedVertexCount;
    ProgressCallback callback = {};
};

}
