#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// returns quasi-uniform 145 samples on unit half-sphere z>0
[[nodiscard]] MRMESH_API std::vector<Vector3f> createSkyPatches();

/// computes the radiation in each sample point by emitting rays from that point in the sky:
/// the radiation is 1.0f if all rays reach the sky not meeting the terrain;
/// the radiation is 0.0f if all rays do not reach the sky because they are intercepted by the terrain
[[nodiscard]] MRMESH_API VertScalars computeSolarRadiation( const Mesh & terrain,
    const VertCoords & samples, const VertBitSet & validSamples );

} //namespace MR
