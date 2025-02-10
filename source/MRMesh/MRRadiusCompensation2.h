#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// position and shape of spherical milling cutter
struct SphericalMillingCutter
{
    Vector3f center;  ///< of spherical part
    float radius = 0; ///< of spherical part
};

/// returns all mesh vertices inside given tool
[[nodiscard]] MRMESH_API VertBitSet findVerticesInsideTool( const Mesh& mesh, const SphericalMillingCutter& tool );

/// structure with parameters for `compensateRadius2` function
struct CompensateRadiusParams2
{
    /// radius of spherical tool
    float toolRadius = 0;

    /// the number of internal iterations, the more iteration - the slower speed but better quality
    int numIters = 10;
};

/// compensate spherical milling tool radius in given mesh making it possible to mill it,
/// returns new vertex positions with same topology
[[nodiscard]] MRMESH_API VertCoords compensateRadius2( const Mesh& mesh, const CompensateRadiusParams2 & params );

} //namespace MR
