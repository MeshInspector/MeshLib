#pragma once

#include "MRObjectMeshData.h"
#include "MRMeshTexture.h"
#include "MRAffineXf3.h"
#include "MRColor.h"
#include <optional>

namespace MR
{

/// ObjectMeshData and additional information from mesh importer
struct LoadedMeshData : ObjectMeshData
{
    MeshTexture texture;
    AffineXf3f xf;
    std::optional<Color> solidColor;
    int skippedFaceCount = 0;
    int duplicatedVertexCount = 0;
};

} //namespace MR
