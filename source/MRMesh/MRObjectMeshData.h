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

/// return all edges separating faces with different colors
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet edgesBetweenDifferentColors( const MeshTopology & topology, const FaceColors & colors );

/// return all edges separating faces with different textures
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet edgesBetweenDifferentTextures( const MeshTopology & topology, const TexturePerFace & textures );

/// resizes each non-empty vertex attribute of data to data.mesh->topology.vertSize() and each non-empty face attribute to faceSize();
/// if an attribute had no values for some valid elements, then it is padded with default values and a warning is logged
MRMESH_API void resizeAttributesToMesh( ObjectMeshData & data );

} //namespace MR
