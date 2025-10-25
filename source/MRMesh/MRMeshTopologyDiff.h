#pragma once

#include "MRMeshTopology.h"
#include "MRphmap.h"

namespace MR
{

/// this object stores a difference between two topologies: both in coordinates and in topology
/// \details if the topologies are similar then this object is small, if the topologies are very distinct then this object will be even larger than one topology itself
/// \ingroup MeshAlgorithmGroup
class MeshTopologyDiff
{
public:
    /// constructs minimal difference, where applyAndSwap( t ) will produce empty topology
    MeshTopologyDiff() = default;

    /// computes the difference, that can be applied to topology-from in order to get topology-to
    MRMESH_API MeshTopologyDiff( const MeshTopology & from, const MeshTopology & to );

    /// given topology-from on input converts it in topology-to,
    /// this object is updated to become the reverse difference from original topology-to to original topology-from
    MRMESH_API void applyAndSwap( MeshTopology & t );

    /// returns true if this object does contain some difference in topology;
    /// if (from) has more topology elements than (to) and the common elements are the same,
    /// then the method will return false since nothing is stored here
    [[nodiscard]] bool any() const { return !changedEdges_.empty(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    size_t toEdgesSize_ = 0;
    HashMap<EdgeId, MeshTopology::HalfEdgeRecord> changedEdges_;
};

} // namespace MR
