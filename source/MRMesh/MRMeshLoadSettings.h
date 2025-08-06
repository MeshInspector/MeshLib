#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// setting for mesh loading from external format, and locations of optional output data
struct MeshLoadSettings
{
    std::optional<Edges>* edges = nullptr; ///< optional load artifact: polyline edges
    VertColors* colors = nullptr;    ///< optional load artifact: per-vertex color map
    VertUVCoords* uvCoords = nullptr;///< optional load artifact: per-vertex uv-coordinates
    VertNormals* normals = nullptr;  ///< optional load artifact: per-vertex normals
    MeshTexture* texture = nullptr;  ///< optional load artifact: texture image
    int* skippedFaceCount = nullptr; ///< optional output: counter of skipped faces during mesh creation
    int* duplicatedVertexCount = nullptr; ///< optional output: counter of duplicated vertices (that created for resolve non-manifold geometry)
    AffineXf3f* xf = nullptr;        ///< optional output: transform for the loaded mesh to improve precision of vertex coordinates
    ProgressCallback callback;       ///< callback for set progress and stop process
};

} //namespace MR
