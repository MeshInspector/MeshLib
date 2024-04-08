#pragma once
#include "exports.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRSolarRadiation.h"

namespace MR
{
namespace Cuda
{

/// computes relative radiation in each valid sample point by emitting rays from that point in the sky:
/// the radiation is 1.0f if all rays reach the sky not hitting the terrain;
/// the radiation is 0.0f if all rays do not reach the sky because they are intercepted by the terrain;
/// \param outSkyRays - optional output bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
///                     0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
/// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
[[nodiscard]] MRCUDA_API VertScalars computeSkyViewFactor( const Mesh& terrain,
    const VertCoords& samples, const VertBitSet& validSamples,
    const std::vector<MR::SkyPatch>& skyPatches,
    BitSet* outSkyRays = nullptr, std::vector<MR::MeshIntersectionResult>* outIntersections = nullptr );

/// In each valid sample point tests the rays from that point in the sky;
/// \return bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
///         0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
/// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
[[nodiscard]] MRCUDA_API BitSet findSkyRays( const Mesh& terrain,
    const VertCoords& samples, const VertBitSet& validSamples,
    const std::vector<MR::SkyPatch>& skyPatches, std::vector<MR::MeshIntersectionResult>* outIntersections = nullptr );
}

}
