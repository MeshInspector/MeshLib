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
    float minAnglePrecision = PI_F / 4.0f; // 20 deg
};

// Returns terrain mesh with structure embedded to it, or error string
// terrain - mesh with +Z normal (not-closed mesh is expected)
// structure - mesh with one open contour and +Z normal, that will be embedded in terrain
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> embedStructureToTerrain( const Mesh& terrain, const Mesh& structure,
    const EmbeddedStructureParameters& params );

// Result structure of fill(mound) and cut(pit) cone creation
struct EmbeddedConeResult
{
    // cone
    Mesh mesh;
    // faces of fill(mound) side 
    FaceBitSet fillBitSet;
    // faces of cut(pit) side
    FaceBitSet cutBitSet;
};

// Parameters structure of fill(mound) and cut(pit) cone creation
struct EmbeddedConeParameters : EmbeddedStructureParameters
{
    // minimum Z plane of cone
    float minZ{ -1.0f };
    // maximum Z plane of cone
    float maxZ{ 1.0f };
    // min angle precision of basin expansion
    float minAnglePrecision = PI_F / 9.0f; // 20 deg
    // input vertices that will have cut(pit) side (inside of terrain)
    // other vertices will have fill(mound) side (outside of terrain)
    VertBitSet cutBitSet;
};

// Returns cone mesh with fill(mound) and cut(pit) sides for embedding in terrain (used in embedStructureToTerrain)
[[nodiscard]] MRMESH_API Expected<EmbeddedConeResult, std::string> createEmbeddedConeMesh( const Contour3f& contour,
    const EmbeddedConeParameters& params );

}