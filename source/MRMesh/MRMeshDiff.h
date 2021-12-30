#pragma once

#include "MRMeshTopology.h"
#include "MRphmap.h"

namespace MR
{

// this object stores a difference between two meshes: both in coordinates and in topology;
// if the meshes are similar then this object is small, if the meshes are very distinct then this object will be comparable to a mesh in size
class MeshDiff
{
public:
    // computes the difference, that can be applied to mesh-from in order to get mesh-to
    MRMESH_API MeshDiff( const Mesh & from, const Mesh & to );

    // given mesh-from on input converts it in mesh-to,
    // this object is updated to become the reverse difference from original mesh-to to original mesh-from
    MRMESH_API void applyAndSwap( Mesh & m );

private:
    size_t toPointsSize_ = 0;
    ParallelHashMap<VertId, Vector3f> changedPoints_;
    size_t toEdgesSize_ = 0;
    ParallelHashMap<EdgeId, MeshTopology::HalfEdgeRecord> changedEdges_;
};

} // namespace MR
