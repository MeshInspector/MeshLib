#pragma once

#include "MRViewerFwd.h"
#include <MRMesh/MRMesh.h>

namespace MR
{

/// result of ObjectMesh subdivision with all new attributes updated
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

    /// moves this result into given object without undo-history,
    /// must be called from main thread
    MRVIEWER_API void assingNoHistory( ObjectMesh& target );

    /// moves this result into given object with undo-history,
    /// must be called from main thread
    MRVIEWER_API void assingWithHistory( const std::shared_ptr<ObjectMesh>& target );
};

struct SubdivideSettings;

/// subdivides given ObjectMesh with given parameters,
/// may be called from parallel thread
[[nodiscard]] MRVIEWER_API ObjectMeshSubdivideResult subdivideObjectMesh( const ObjectMesh& obj, const SubdivideSettings& subs );

} //namespace MR
