#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <iostream>
#include <optional>

namespace MR
{

/// optional load artifacts and other setting for PLY file loading
struct PlyLoadParams
{
    std::optional<Triangulation>* tris = nullptr; ///< optional load artifact: mesh triangles
    std::optional<Edges>* edges = nullptr; ///< optional load artifact: polyline edges
    VertColors* colors = nullptr;    ///< optional load artifact: per-vertex color map
    VertUVCoords* uvCoords = nullptr;///< optional load artifact: per-vertex uv-coordinates
    VertNormals* normals = nullptr;  ///< optional load artifact: per-vertex normals
    MeshTexture* texture = nullptr;  ///< optional load artifact: texture image
    ProgressCallback callback;       ///< callback for set progress and stop process
};

[[nodiscard]] MRMESH_API Expected<VertCoords> loadPly( std::istream& in, const PlyLoadParams& params );

} //namespace MR
