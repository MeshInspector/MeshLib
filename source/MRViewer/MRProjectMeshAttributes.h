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

/// finds attributes of new mesh part by projecting region's faces/vertices on old mesh
/// returns nullopt if canceled by progress bar
[[nodiscard]] MRVIEWER_API std::optional<MeshAttributes> projectMeshAttributes(
    const ObjectMesh& oldMeshObj,
    const MeshPart& newMeshPart,
    ProgressCallback cb = {} );

/// set new mesh attributes and saving the history of changing mesh attributes
MRVIEWER_API void emplaceMeshAttributes(
    std::shared_ptr<ObjectMesh> objectMesh,
    MeshAttributes&& newAttribute );

}
