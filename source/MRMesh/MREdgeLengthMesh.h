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

    /// computes cotangent of the angle in the left( e ) triangle opposite to e,
    /// and returns 0 if left face does not exist
    [[nodiscard]] MRMESH_API float leftCotan( EdgeId e ) const;

    /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
    /// consider cotangents zero for not existing triangles
    [[nodiscard]] float cotan( UndirectedEdgeId ue ) const { EdgeId e{ ue }; return leftCotan( e ) + leftCotan( e.sym() ); }

    /// returns true if given edge satisfies Delaunay conditions,
    /// returns false if the edge needs to be flipped to satisfy Delaunay conditions,
    /// passing negative threshold makes more edges satisfy Delaunay conditions
    [[nodiscard]] bool isDelone( UndirectedEdgeId ue, float threshold = 0 ) const { return cotan( ue ) >= threshold; }
};

} //namespace MR
