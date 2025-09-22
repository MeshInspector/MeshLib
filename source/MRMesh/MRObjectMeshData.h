#pragma once

#include "MRColor.h"
#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRBitSet.h"

namespace MR
{

/// mesh and its per-element attributes for ObjectMeshHolder
struct ObjectMeshData
{
    std::shared_ptr<Mesh> mesh;

    // selection
    FaceBitSet selectedFaces;
    UndirectedEdgeBitSet selectedEdges;

    UndirectedEdgeBitSet creases;

    // colors
    VertColors vertColors;
    FaceColors faceColors;

    // textures
    VertUVCoords uvCoordinates; ///< vertices coordinates in texture
    TexturePerFace texturePerFace;

    /// returns copy of this object with mesh cloned
    [[nodiscard]] MRMESH_API ObjectMeshData clone() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;
};

} //namespace MR
