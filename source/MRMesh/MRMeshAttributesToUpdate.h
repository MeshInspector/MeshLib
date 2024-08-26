#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// the attribute data of the mesh that needs to be updated
struct MeshAttributesToUpdate
{
    VertUVCoords* uvCoords = nullptr;
    VertColors* colorMap = nullptr;

    TexturePerFace* texturePerFace = nullptr;
    FaceColors* faceColors = nullptr;
};
}
