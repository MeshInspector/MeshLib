#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRExpected.h"

namespace MR
{

// Parameters of structure embedding in terrain
struct EmbeddedStructureParameters
{
    // angle of fill cone (mound)
    float fillAngle = 0.0f;
    // angle of cut cone (pit)
    float cutAngle = 0.0f;
    // min angle precision of basin expansion
    float minAnglePrecision = PI_F / 9.0f; // 20 deg
    // optional out new faces of embedded structure 
    FaceBitSet* outStructFaces{ nullptr };
    // optional out new faces of fill part
    FaceBitSet* outFillFaces{ nullptr };
    // optional out new faces of cut part
    FaceBitSet* outCutFaces{ nullptr };
    // optional out map new terrain faces to old terrain faces
    FaceMap* new2oldFaces{ nullptr };
};

// Returns terrain mesh with structure embedded to it, or error string
// terrain - mesh with +Z normal (not-closed mesh is expected)
// structure - mesh with one open contour and +Z normal, that will be embedded in terrain
[[nodiscard]] MRMESH_API Expected<Mesh> embedStructureToTerrain( const Mesh& terrain, const Mesh& structure,
    const EmbeddedStructureParameters& params );

}