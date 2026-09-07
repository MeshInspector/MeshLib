#pragma once

#include "MRObjectMeshData.h"
#include "MRMeshTexture.h"
#include "MRAffineXf3.h"

namespace MR
{

/// ObjectMeshData and additional information from mesh importer
struct LoadedMeshData : ObjectMeshData
{
    MeshTexture texture;
    AffineXf3f xf;
    int skippedFaceCount = 0;
    int duplicatedVertexCount = 0;
};

} //namespace MR
