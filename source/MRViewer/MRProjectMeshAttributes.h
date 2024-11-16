#pragma once
#include "MRViewerFwd.h"
#include <MRMesh/MRMeshFwd.h>
#include "MRMesh/MRVector.h"

#include <optional>

namespace MR
{

struct MeshAttributes
{
    VertUVCoords uvCoords;
    VertColors colorMap;

    TexturePerFace texturePerFace;
    FaceColors faceColors;
};

// projecting the attributes of the old mesh onto the new
// returns nullopt if canceled by progress bar
[[nodiscard]] MRVIEWER_API std::optional<MeshAttributes> projectMeshAttributes(
    const ObjectMesh& oldMesh,
    const MeshPart& mp,
    ProgressCallback cb = {} );

// set new mesh attributes and saving the history of changing mesh attributes
MRVIEWER_API void emplaceMeshAttributes(
    std::shared_ptr<ObjectMesh> objectMesh,
    MeshAttributes&& newAttribute );

}
