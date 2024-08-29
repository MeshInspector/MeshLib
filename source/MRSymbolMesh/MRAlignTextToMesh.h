#pragma once

#include "MRSymbolMeshFwd.h"
#include "MRSymbolMesh.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMeshTriPoint.h"

namespace MR
{

struct TextMeshAlignParams : SymbolMeshParams
{
    // Start coordinate on mesh, represent lowest left corner of text
    MeshTriPoint startPoint;
    // Position of the startPoint in a text bounding box
    // (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotPoint{0.0f, 0.0f};
    // Direction of text
    Vector3f direction;
    // Text normal to surface, if nullptr - use mesh normal at `startPoint`
    const Vector3f* textNormal{nullptr};
    // Font height, meters
    float fontHeight{1.0f};
    // Text mesh inside and outside offset of input mesh
    float surfaceOffset{1.0f};
    // Maximum possible movement of text mesh alignment, meters
    float textMaximumMovement{2.5f};
    // If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text
    bool fontBasedSizeCalc{ false };
};

// Creates symbol mesh and aligns it to given surface
MRSYMBOLMESH_API Expected<Mesh> alignTextToMesh( const Mesh& mesh, const TextMeshAlignParams& params );
}
