#pragma once

#include "MRObjectMeshData.h"
#include "MRAffineXf3.h"

namespace MR
{

/// ObjectMeshData and additional information from mesh importer
struct LoadedMeshData : ObjectMeshData
{
    int skippedFaceCount = 0;
    int duplicatedVertexCount = 0;
    AffineXf3f xf;
};

} //namespace MR
