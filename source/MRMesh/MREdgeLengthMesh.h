#pragma once

#include "MRMeshTopology.h"

namespace MR
{

/// Unlike the classic mesh that stores coordinates of its vertices, this class
/// stores the lengths of all edges. It can be used for construction of intrinsic Intrinsic Delaunay Triangulations.
/// \ingroup MeshGroup
struct [[nodiscard]] EdgeLengthMesh
{
    MeshTopology topology;
    UndirectedEdgeScalars edgeLengths;

    /// construct EdgeLengthMesh from an ordinary mesh
    [[nodiscard]] MRMESH_API static EdgeLengthMesh fromMesh( const Mesh& mesh );
};

} //namespace MR
