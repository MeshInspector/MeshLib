#pragma once

#include "MRMeshTopology.h"
#include "MRphmap.h"

namespace MR
{

/// this object stores a difference between two meshes: both in coordinates and in topology
/// \details if the meshes are similar then this object is small, if the meshes are very distinct then this object will be comparable to a mesh in size
/// \ingroup MeshAlgorithmGroup
class MeshDiff
{
public:
    /// computes the difference, that can be applied to mesh-from in order to get mesh-to
    MRMESH_API MeshDiff( const Mesh & from, const Mesh & to );

    /// given mesh-from on input converts it in mesh-to,
    /// this object is updated to become the reverse difference from original mesh-to to original mesh-from
    MRMESH_API void applyAndSwap( Mesh & m );

    /// returns true if this object does contain some difference in point coordinates or in topology;
    /// if (from) mesh has just more points or more topology elements than (to) and the common elements are the same,
    /// then the method will return false since nothing is stored here
    [[nodiscard]] bool any() const { return !changedPoints_.empty() || !changedEdges_.empty(); }

private:
    size_t toPointsSize_ = 0;
    ParallelHashMap<VertId, Vector3f> changedPoints_;
    size_t toEdgesSize_ = 0;
    ParallelHashMap<EdgeId, MeshTopology::HalfEdgeRecord> changedEdges_;
};

} // namespace MR
