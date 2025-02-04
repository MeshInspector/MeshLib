#pragma once


#include "MRMeshFwd.h"

#include "MRExpected.h"
#include "MRMesh.h"

namespace MR
{

struct NoiseSettings
{
    float sigma = 0.01f;
    // start state of the generator engine
    unsigned int seed = 0;
    ProgressCallback callback = {};
};

// Adds noise to the points, using a normal distribution
MRMESH_API Expected<void> addNoise( VertCoords& points, const VertBitSet& validVerts, NoiseSettings settings );
inline Expected<void> addNoise( Mesh& mesh, const VertBitSet* region = nullptr, NoiseSettings settings = {} )
{
    return addNoise( mesh.points, mesh.topology.getVertIds( region ), settings );
}

}
