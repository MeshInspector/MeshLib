#pragma once


#include "MRMeshFwd.h"

namespace MR
{
// Adds noise to the points, using a normal distribution
// seed - start state of the generator engine
MRMESH_API void addNoise( VertCoords& points, const VertBitSet& validVerts, float sigma, unsigned int seed );

}
