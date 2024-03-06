#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// returns quasi-uniform 145 samples on unit half-sphere z>0
[[nodiscard]] MRMESH_API std::vector<Vector3f> sampleHalfSphere();

/// this class represents a portion of the sky, and its radiation
struct SkyPatch
{
    /// direction toward the center of the patch
    Vector3f dir;
    /// radiation of the patch depending on Sun's position, sky clearness and brightness, etc
    float radiation = 0;
};

/// computes relative radiation in each valid sample point by emitting rays from that point in the sky:
/// the radiation is 1.0f if all rays reach the sky not hitting the terrain;
/// the radiation is 0.0f if all rays do not reach the sky because they are intercepted by the terrain;
/// \param outSkyRays - optional output bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
///                     0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
/// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
[[nodiscard]] MRMESH_API VertScalars computeSkyViewFactor( const Mesh & terrain,
    const VertCoords & samples, const VertBitSet & validSamples,
    const std::vector<SkyPatch> & skyPatches,
    BitSet * outSkyRays = nullptr, std::vector<MeshIntersectionResult>* outIntersections = nullptr );

/// In each valid sample point tests the rays from that point in the sky;
/// \return bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
///         0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
/// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
[[nodiscard]] MRMESH_API BitSet findSkyRays( const Mesh & terrain,
    const VertCoords & samples, const VertBitSet & validSamples,
    const std::vector<SkyPatch> & skyPatches, std::vector<MeshIntersectionResult>* outIntersections = nullptr );

} //namespace MR
