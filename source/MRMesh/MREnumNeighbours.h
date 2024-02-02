#pragma once

#include "MRMeshFwd.h"
#include "MRBitSet.h"

namespace MR
{

class EnumNeihbourVertices
{
public:
    /// invokes given predicate for vertices starting from \param start,
    /// and continuing to all immediate neighbours in depth-first order until the predicate returns false
    MRMESH_API void run( const MeshTopology & topology, VertId start, const VertPredicate & pred );

private:
    VertBitSet visited_;
    std::vector<VertId> bd_;
};

/// computes Euclidean 3D distances from given start point to all neighbor vertices within given \param range
/// and to first vertices with the distance more or equal than range
[[nodiscard]] MRMESH_API VertScalars computeSpaceDistances( const Mesh& mesh, const PointOnFace & start, float range );

/// calculates all neighbor vertices within a given \param range
/// and to first vertices with the distance more or equal than range
[[nodiscard]] MRMESH_API VertBitSet findNeighborVerts( const Mesh& mesh, const PointOnFace& start, float range );


class EnumNeihbourFaces
{
public:
    /// invokes given predicate for faces starting from all incident to \param start,
    /// and continuing to all immediate neighbours in depth-first order until the predicate returns false
    MRMESH_API void run( const MeshTopology & topology, VertId start, const FacePredicate & pred );

private:
    FaceBitSet visited_;
    std::vector<FaceId> bd_;
};


} //namespace MR
