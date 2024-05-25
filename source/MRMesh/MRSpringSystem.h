#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct SpringSystemSettings
{
    /// vertices to be moved by the algorithm, nullptr means all valid vertices
    const VertBitSet* region = nullptr;

    /// target distance of each edge in the mesh (for at least one edge's vertex in the region)
    UndirectedEdgeMetric springRestLength; // must be defined by the caller

    /// the algorithm is iterative, the more iterations the closer result to exact solution
    int numIters = 5;
};

/// Moves given vertices to make the distances between them as specified
MRMESH_API void solveSpringSystem( Mesh& mesh, const SpringSystemSettings & settings );

[[nodiscard]] MRMESH_API float springDiscrepancyAtVertex( const Mesh & mesh, VertId v, const Vector3f & vpos, const UndirectedEdgeMetric & springRestLength );

[[nodiscard]] MRMESH_API Vector3f springBestVertexPos( const Mesh & mesh, VertId v, const UndirectedEdgeMetric & springRestLength );

} //namespace MR
