#pragma once


#include "MRPch/MRTBB.h"
#include "MRMesh/MRBitSet.h"
#include "MRMeshFwd.h"

#include <random>

namespace MR
{
// Adds noise to the points, using a normal distribution
// seed - start state of the random device engine
MRMESH_API void addNoise( VertCoords& points, const VertBitSet& validVerts, float sigma, unsigned int seed );

}
