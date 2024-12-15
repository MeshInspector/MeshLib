#pragma once

#include "MRphmap.h"

namespace MR
{

/// this object stores a difference between two vectors with 3D coordinates
/// \details if the vectors are similar then this object is small, if the vectors are very distinct then this object will be even larger than one vector itself
/// \ingroup MeshAlgorithmGroup
class VertCoordsDiff
{
public:
    /// constructs minimal difference, where applyAndSwap( v ) will produce empty vector
    VertCoordsDiff() = default;

    /// computes the difference, that can be applied to vector-from in order to get vector-to
    MRMESH_API VertCoordsDiff( const VertCoords & from, const VertCoords & to );

    /// given vector-from on input converts it in vector-to,
    /// this object is updated to become the reverse difference from original vector-to to original vector-from
    MRMESH_API void applyAndSwap( VertCoords & m );

    /// returns true if this object does contain some difference in point coordinates;
    /// if (from) vector has just more points and the common elements are the same,
    /// then the method will return false since nothing is stored here
    [[nodiscard]] bool any() const { return !changedPoints_.empty(); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    size_t toPointsSize_ = 0;
    HashMap<VertId, Vector3f> changedPoints_;
};

} // namespace MR
