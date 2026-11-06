#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

template<typename T> struct BoxedLeaf;

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
    using LeafBoxes = Vector<BoxT, LeafId>;

public:
    /// const-access to all not-leaf nodes, a tree with N leaves has N-1 nodes
    [[nodiscard]] const NodeVec & nodes() const { return nodes_; }

    /// const-access to any not-leaf node
    [[nodiscard]] const Node & operator[]( NodeId nid ) const { return nodes_[nid]; }

    /// bounding boxes of all leaves, indexed by leaf id; the boxes of the ids not in the tree are not specified
    [[nodiscard]] const LeafBoxes & leafBoxes() const { return leafBoxes_; }

    /// returns true if given node id is in fact a leaf (encoded as negative number)
    [[nodiscard]] static bool isLeaf( NodeId n ) { return Node::isLeaf( n ); }

    /// returns the id of the leaf encoded in given node id
    [[nodiscard]] static LeafId leafId( NodeId n ) { return Node::leafId( n ); }

    /// returns the bounding box of given node or leaf
    [[nodiscard]] const BoxT & box( NodeId n ) const { return isLeaf( n ) ? leafBoxes_[leafId( n )] : nodes_[n].box; }

    /// returns true if the tree has no leaves
    [[nodiscard]] bool empty() const { return leafBoxes_.empty(); }

    /// returns root node id: 0 if the tree has at least two leaves, or the only leaf;
    /// must not be called for empty tree
    [[nodiscard]] NodeId rootNodeId() const { assert( !empty() ); return root_; }

    /// returns the root node bounding box
    [[nodiscard]] BoxT getBoundingBox() const { return empty() ? BoxT{} : box( root_ ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return nodes_.heapBytes() + leafBoxes_.heapBytes(); }

    /// returns the number of leaves in whole tree
    [[nodiscard]] size_t numLeaves() const { return empty() ? 0 : nodes_.size() + 1; }

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
    /// builds the tree from given leaves with their boxes; leafIdSize is one more than the maximal leaf id
    void build_( Buffer<BoxedLeaf<T>> boxedLeaves, size_t leafIdSize );

    NodeVec nodes_;
    LeafBoxes leafBoxes_;
    NodeId root_;
};

} //namespace MR
