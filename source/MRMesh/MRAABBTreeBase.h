#pragma once

#include "MRAABBTreeNode.h"

namespace MR
{

/// base class for most AABB-trees (except for AABBTreePoints)
template <typename T>
class AABBTreeBase
{
public:
    using Traits = T;
    using Node = AABBTreeNode<Traits>;
    using NodeVec = Vector<Node, NodeId>;

public:
    /// const-access to all nodes
    [[nodiscard]] const NodeVec & nodes() const { return nodes_; }

    /// const-access to any node
    [[nodiscard]] const Node & operator[]( NodeId nid ) const { return nodes_[nid]; }

    /// returns root node id
    [[nodiscard]] static NodeId rootNodeId() { return NodeId{ 0 }; }

    /// returns the root node bounding box
    [[nodiscard]] auto getBoundingBox() const { return nodes_.empty() ? typename Node::BoxT{} : nodes_[rootNodeId()].box; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const { return nodes_.heapBytes(); }

    /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;

protected:
    NodeVec nodes_;
};

} //namespace MR
