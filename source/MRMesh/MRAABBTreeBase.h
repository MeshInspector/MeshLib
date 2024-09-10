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

public:
    /// const-access to all nodes
    [[nodiscard]] const NodeVec & nodes() const { return nodes_; }

    /// const-access to any node
    [[nodiscard]] const Node & operator[]( NodeId nid ) const { return nodes_[nid]; }

    /// returns root node id
    [[nodiscard]] static NodeId rootNodeId() { return NodeId{ 0 }; }

    /// returns the root node bounding box
    [[nodiscard]] BoxT getBoundingBox() const { return nodes_.empty() ? BoxT{} : nodes_[rootNodeId()].box; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return nodes_.heapBytes(); }

    /// returns the number of leaves in whole tree
    [[nodiscard]] size_t numLeaves() const { return nodes_.empty() ? 0 : ( nodes_.size() + 1 ) / 2; }

    /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;

    /// returns all leaves in the subtree with given root
    [[nodiscard]] MRMESH_API LeafBitSet getSubtreeLeaves( NodeId subtreeRoot ) const;

    /// returns set of nodes containing among direct or indirect children given leaves
    [[nodiscard]] MRMESH_API NodeBitSet getNodesFromLeaves( const LeafBitSet & leaves ) const;

    /// fills map: LeafId -> leaf#;
    /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
    MRMESH_API void getLeafOrder( LeafBMap & leafMap ) const;

    /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
    /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
    MRMESH_API void getLeafOrderAndReset( LeafBMap & leafMap );

protected:
    NodeVec nodes_;
};

} //namespace MR
