#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

struct ReduceTotalAngleParams
{
    /// Maximal allowed dihedral angle change (in radians) over the flipped edge
    float maxAngleChange = FLT_MAX;

    /// if this value is less than FLT_MAX then the algorithm will
    /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
    float criticalTriAspectRatio = FLT_MAX;

    /// This value must be in [0,1] range;
    /// factorDelone = 0 means that only dihedral angles are minimized, ignoring Delaunay criterion;
    /// factorDelone = 1 means that only Delaunay criterion is optimized ignoring dihedral angles;
    /// other values mean that a mixture of both criteria will be optimized
    float factorDelone = 0.1f;

    /// Only edges with left and right faces in this set can be flipped
    const FaceBitSet* region = nullptr;

    /// Edges specified by this bit-set will never be flipped
    const UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Only edges with origin or destination in this set before or after flip can be flipped
    const VertBitSet* vertRegion = nullptr;
};

/// minimizes summed deviation of triangle-triangle angles from plane, where each edge is weighted by its length;
/// performs the given number of iterations, during each the edges are flipped;
/// returns the total number of flips done
MRMESH_API int reduceTotalAngle( MeshTopology& topology, const VertCoords& points, int numIters,
    const ReduceTotalAngleParams& params = {}, const ProgressCallback& progressCallback = {} );
MRMESH_API int reduceTotalAngleInMesh( Mesh& mesh, int numIters,
    const ReduceTotalAngleParams& params = {}, const ProgressCallback& progressCallback = {} );

} //namespace MR
