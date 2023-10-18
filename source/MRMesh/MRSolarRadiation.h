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

/// computes relative radiation in each sample point by emitting rays from that point in the sky:
/// the radiation is 1.0f if all rays reach the sky not meeting the terrain;
/// the radiation is 0.0f if all rays do not reach the sky because they are intercepted by the terrain
[[nodiscard]] MRMESH_API VertScalars computeSkyViewFactor( const Mesh & terrain,
    const VertCoords & samples, const VertBitSet & validSamples,
    const std::vector<SkyPatch> & skyPatches );

} //namespace MR
