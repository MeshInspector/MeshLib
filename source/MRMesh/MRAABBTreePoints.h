#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector.h"
#include "MRVector3.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// bounding volume hierarchy for point cloud structure
class AABBTreePoints
{
public:
    struct Node
    {
        Box3f box; ///< bounding box of whole subtree
        NodeId l;  ///< left child node for an inner node, or -(l+1) is the index of the first point in a leaf node
        NodeId r;  ///< right child node for an inner node, or -(r+1) is the index of the last point in a leaf node
        /// returns true if node represent real points, false if it has child nodes
        bool leaf() const { return !l.valid(); }
        /// returns [first,last) indices of leaf points
        std::pair<int, int> getLeafPointRange() const { assert( leaf() ); return { -( l + 1 ),-( r + 1 ) }; }
        /// sets [first,last) to this node (leaf)
        void setLeafPointRange( int first, int last ) { l = NodeId( -( first + 1 ) ); r = NodeId( -( last + 1 ) ); }
    };
    using NodeVec = Vector<Node, NodeId>;
    using NodeBitSet = TaggedBitSet<NodeTag>;
    [[nodiscard]] const NodeVec& nodes() const { return nodes_; }
    [[nodiscard]] const Node& operator[]( NodeId nid ) const { return nodes_[nid]; }
    [[nodiscard]] static NodeId rootNodeId() { return NodeId{0}; }
    /// returns the root node bounding box
    [[nodiscard]] Box3f getBoundingBox() const { return nodes_.empty() ? Box3f{} : nodes_[rootNodeId()].box; }

    struct Point
    {
        Vector3f coord;
        VertId id;
    };
    [[nodiscard]] const std::vector<Point>& orderedPoints() const { return orderedPoints_; }

    /// creates tree for given point cloud
    MRMESH_API AABBTreePoints( const PointCloud& pointCloud );
    /// creates tree for vertices of given mesh
    MRMESH_API AABBTreePoints( const Mesh& mesh );
    /// creates tree from given valid points
    MRMESH_API AABBTreePoints( const VertCoords & points, const VertBitSet * validPoints = nullptr );
    /// creates tree from given valid points
    AABBTreePoints( const VertCoords & points, const VertBitSet & validPoints ) : AABBTreePoints( points, &validPoints ) {}

    /// maximum number of points in leaf node of tree (all of leafs should have this number of points except last one)
    constexpr static int MaxNumPointsInLeaf = 16;

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
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit)
    MRMESH_API void refit( const VertCoords & newCoords, const VertBitSet & changedVerts );

private:
    std::vector<Point> orderedPoints_;
    NodeVec nodes_;

    AABBTreePoints( const AABBTreePoints & ) = default;
    AABBTreePoints & operator =( const AABBTreePoints & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreePoints>;
    friend class SharedThreadSafeOwner<AABBTreePoints>;
};

// returns the number of nodes in the binary tree with given number of points
inline int getNumNodesPoints( int numPoints )
{
    assert( numPoints > 0 );
    return 2 * ( ( numPoints + AABBTreePoints::MaxNumPointsInLeaf - 1 ) / AABBTreePoints::MaxNumPointsInLeaf ) - 1;
}

/// \}

} // namespace MR
