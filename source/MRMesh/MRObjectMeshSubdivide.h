#pragma once

#include "MRMesh.h"

namespace MR
{

struct ObjectMeshSubdivideResult
{
    Mesh mesh;
    FaceBitSet selFaces;
    UndirectedEdgeBitSet selEdges;
    UndirectedEdgeBitSet creases;
    VertUVCoords uvCoords;
    VertColors colorMap;
    TexturePerFace texturePerFace;
    FaceColors faceColors;

    MRMESH_API void assingNoHistory( ObjectMesh & target );
};

struct SubdivideSettings;

[[nodiscard]] MRMESH_API ObjectMeshSubdivideResult subdivideObjectMesh( const ObjectMesh& obj, const SubdivideSettings& subs );

} //namespace MR
