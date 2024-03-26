#pragma once


#include "MRPch/MRTBB.h"
#include "MRMesh/MRBitSet.h"
#include "MRMeshFwd.h"

#include <random>

namespace MR
{
// Adds noise to the points, using a normal distribution with a starting value
MRMESH_API void addNoise( VertCoords& points, const VertBitSet& validVerts, float sigma, unsigned int startValue );

}
