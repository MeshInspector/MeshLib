#pragma once

#include "MRVertCoordsDiff.h"
#include "MRMeshTopologyDiff.h"

namespace MR
{

/// this object stores a difference between two meshes: both in coordinates and in topology
/// \details if the meshes are similar then this object is small, if the meshes are very distinct then this object will be even larger than one mesh itself
/// \ingroup MeshAlgorithmGroup
class MeshDiff
{
public:
    /// constructs minimal difference, where applyAndSwap( m ) will produce empty mesh
    MeshDiff() = default;

    /// computes the difference, that can be applied to mesh-from in order to get mesh-to
    MRMESH_API MeshDiff( const Mesh & from, const Mesh & to );

    /// given mesh-from on input converts it in mesh-to,
    /// this object is updated to become the reverse difference from original mesh-to to original mesh-from
    MRMESH_API void applyAndSwap( Mesh & m );

    /// returns true if this object does contain some difference in point coordinates or in topology;
    /// if (from) mesh has just more points or more topology elements than (to) and the common elements are the same,
    /// then the method will return false since nothing is stored here
    [[nodiscard]] bool any() const { return pointsDiff_.any() || topologyDiff_.any(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return pointsDiff_.heapBytes() + topologyDiff_.heapBytes(); }

private:
    VertCoordsDiff pointsDiff_;
    MeshTopologyDiff topologyDiff_;
};

} // namespace MR
