#pragma once

#include "MRAABBTreeSimd.h"
#include "MRBox.h"
#include "MRId.h"
#include "MRVector.h"
#include "MRVector3.h"
#include <algorithm>
#include <utility>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// bounding volume hierarchy of a point cloud with up to 4 children per node in SIMD layout: every internal node keeps
/// the boxes of its children; the points of leaf l are orderedPoints() [MaxNumPointsInLeaf * l, MaxNumPointsInLeaf * ( l + 1 ) ),
/// all leaves are full except the last one
class AABBTreePoints
{
public:
    static constexpr int MaxChildren = 4;
    /// maximum number of points in leaf node of tree (all of leafs should have this number of points except last one)
    constexpr static int MaxNumPointsInLeaf = 16;
    /// the deepest level of a node (the root is at 0): the stacks of the traversals are sized for it
    static constexpr int MaxDepth = 16;

    using SimdNode = AABBTreeSimdNode;

    struct Point
    {
        Vector3f coord;
        VertId id;
    };
    [[nodiscard]] const std::vector<Point>& orderedPoints() const { return orderedPoints_; }

    /// returns true if the tree has no points
    [[nodiscard]] bool empty() const { return orderedPoints_.empty(); }

    /// the internal nodes, the root is the first one (if the tree has more than MaxNumPointsInLeaf points); the children follow their parent
    [[nodiscard]] MR_BIND_IGNORE const Vector<SimdNode> & simdNodes() const { return simdNodes_; }

    /// the number of leaves
    [[nodiscard]] int numLeaves() const { return ( int( orderedPoints_.size() ) + MaxNumPointsInLeaf - 1 ) / MaxNumPointsInLeaf; }

    /// returns [first,last) indices of the points of given leaf in orderedPoints()
    [[nodiscard]] std::pair<int, int> leafPointRange( int leaf ) const
        { return { leaf * MaxNumPointsInLeaf, std::min( ( leaf + 1 ) * MaxNumPointsInLeaf, int( orderedPoints_.size() ) ) }; }

    /// the reference to the root as in SimdNode::child: 0, or ~0 if the tree has one leaf only
    [[nodiscard]] int rootRef() const { return simdNodes_.empty() ? ~0 : 0; }

    /// returns the bounding box of all points
    [[nodiscard]] Box3f getBoundingBox() const { return box_; }

    /// creates tree for given point cloud
    MRMESH_API AABBTreePoints( const PointCloud& pointCloud );
    /// creates tree for vertices of given mesh
    MRMESH_API AABBTreePoints( const Mesh& mesh );
    /// creates tree from given valid points
    MRMESH_API AABBTreePoints( const VertCoords & points, const VertBitSet * validPoints = nullptr );

    /// creates tree from given valid points
    /// This is skipped in the bindings because it conflicts with the overload taking a pointer in C#. Since that overload is strictly more useful, we're keeping that one.
    MR_BIND_IGNORE AABBTreePoints( const VertCoords & points, const VertBitSet & validPoints ) : AABBTreePoints( points, &validPoints ) {}

    AABBTreePoints( AABBTreePoints && ) noexcept = default;
    AABBTreePoints & operator =( AABBTreePoints && ) noexcept = default;

    /// returns the mapping original VertId to new id following the points order in the tree;
    /// buffer in vertMap must be resized before the call, and caller is responsible for filling missing vertex elements
    MRMESH_API void getLeafOrder( VertBMap & vertMap ) const;
    /// returns the mapping original VertId to new id following the points order in the tree;
    /// then resets leaf order as if the points were renumberd following the mapping;
    /// buffer in vertMap must be resized before the call, and caller is responsible for filling missing vertex elements
    MRMESH_API void getLeafOrderAndReset( VertBMap & vertMap );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

    /// updates bounding boxes of the nodes containing changed vertices;
    /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
    /// \param newCoords coordinates of all vertices including changed ones;
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit), nullptr means all vertices
    MRMESH_API void refit( const VertCoords & newCoords, const VertBitSet * changedVerts );

private:
    std::vector<Point> orderedPoints_;
    Vector<SimdNode> simdNodes_;
    Box3f box_;

    AABBTreePoints( const AABBTreePoints & ) = default;
    AABBTreePoints & operator =( const AABBTreePoints & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreePoints>;
    friend class SharedThreadSafeOwner<AABBTreePoints>;
};

/// \}

} // namespace MR
