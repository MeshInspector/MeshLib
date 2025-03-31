#pragma once

#include "MRMeshTopology.h"
#include <optional>

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

    /// given the edge with left and right triangular faces, which form together a quadrangle,
    /// returns the length of geodesic line on original mesh between the vertices of the quadrangle opposite to given edge;
    /// returns std::nullopt if the geodesic line does not go fully inside the quadrangle
    [[nodiscard]] std::optional<float> edgeLengthAfterFlip( EdgeId e ) const;

    /// given the edge with left and right triangular faces, which form together a quadrangle,
    /// rotates the edge counter-clockwise inside the quadrangle;
    /// the length of e becomes equal to the length of geodesic line between its new ends on original mesh;
    /// does not flip and returns false if the geodesic line does not go fully inside the quadrangle
    MRMESH_API bool flipEdge( EdgeId e );
};

} //namespace MR
