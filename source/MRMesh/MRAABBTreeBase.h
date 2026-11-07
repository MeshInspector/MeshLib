#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

/// base class for most AABB-trees (except for AABBTreePoints)
template <typename T>
class MRMESH_CLASS AABBTreeBase
{
public:
    using Traits = T;
    using Node = AABBTreeNode<Traits>;
    using NodeVec = Vector<Node, NodeId>;
    using LeafTag = typename T::LeafTag;
    using LeafId = typename T::LeafId;
    using LeafBitSet = TaggedBitSet<LeafTag>;
    using LeafBMap = BMap<LeafId, LeafId>;
    using BoxT = typename T::BoxT;
    using Leaf = typename AABBTreeLeaf<T>::type;
    using Leaves = Vector<Leaf>;
    /// whether the boxes of leaves are stored in the tree (all trees except AABBTree of a mesh)
    static constexpr bool leavesHaveBoxes = !std::is_same_v<Leaf, FaceLeaf>;

public:
    /// const-access to all not-leaf nodes, a tree with N leaves has N-1 nodes
    [[nodiscard]] const NodeVec & nodes() const { return nodes_; }

    /// const-access to any not-leaf node
    [[nodiscard]] const Node & operator[]( NodeId nid ) const { return nodes_[nid]; }

    /// all leaves in the order of the tree right after its construction
    [[nodiscard]] const Leaves & leaves() const { return leaves_; }

    /// returns true if given node id is in fact a leaf (encoded as negative number)
    [[nodiscard]] static bool isLeaf( NodeId n ) { return Node::isLeaf( n ); }

    /// returns the leaf encoded in given node id
    [[nodiscard]] const Leaf & leaf( NodeId n ) const { return leaves_[Node::leafIndex( n )]; }

    /// returns the id of the leaf encoded in given node id
    [[nodiscard]] LeafId leafId( NodeId n ) const { return leaf( n ).leafId; }

    /// returns the bounding box of given node or leaf
    [[nodiscard]] const BoxT & box( NodeId n ) const requires leavesHaveBoxes { return isLeaf( n ) ? leaf( n ).box : nodes_[n].box; }

    /// returns true if the tree has no leaves
    [[nodiscard]] bool empty() const { return leaves_.empty(); }

    /// returns root node id: 0 if the tree has at least two leaves, or the only leaf;
    /// must not be called for empty tree
    [[nodiscard]] NodeId rootNodeId() const { assert( !empty() ); return nodes_.empty() ? Node::leafChild( 0 ) : NodeId( 0 ); }

    /// returns the root node bounding box
    [[nodiscard]] BoxT getBoundingBox() const requires leavesHaveBoxes { return empty() ? BoxT{} : box( rootNodeId() ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return nodes_.heapBytes() + leaves_.capacity() * sizeof( Leaf ); }

    /// returns the number of leaves in whole tree
    [[nodiscard]] size_t numLeaves() const { return leaves_.size(); }

    /// returns at least given number of top-level not-intersecting subtrees (some can be leaves), union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;

    /// returns all leaves in the subtree with given root (can be a leaf)
    [[nodiscard]] MRMESH_API LeafBitSet getSubtreeLeaves( NodeId subtreeRoot ) const;

    /// returns set of not-leaf nodes containing among direct or indirect children given leaves
    [[nodiscard]] MRMESH_API NodeBitSet getNodesFromLeaves( const LeafBitSet & leaves ) const;

    /// fills map: LeafId -> leaf#;
    /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
    MRMESH_API void getLeafOrder( LeafBMap & leafMap ) const;

    /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
    /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
    MRMESH_API void getLeafOrderAndReset( LeafBMap & leafMap );

protected:
    /// builds the tree from given leaves with their boxes
    void build_( Vector<BoxedLeaf<T>> leaves ) requires leavesHaveBoxes;

    NodeVec nodes_;
    Leaves leaves_;
};

} //namespace MR
