#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRExpected.h"

namespace MR
{

struct EmbeddedConeResult
{
    Mesh mesh;
    FaceBitSet fillBitSet;
    FaceBitSet cutBitSet;
};

struct EmbeddedConeParameters
{
    float fillAngle = 0.0f;
    float cutAngle = 0.0f;
    float minZ{ -1.0f };
    float maxZ{ 1.0f };
    float pixelSize{ 1.0f };
    VertBitSet cutBitSet;
};

[[nodiscard]] MRMESH_API Expected<EmbeddedConeResult, std::string> createEmbeddedConeMesh( const Contour3f& contour,
    const EmbeddedConeParameters& params );





}