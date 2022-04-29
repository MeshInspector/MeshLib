#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// @{

template<typename V>
struct PolylineTraits;

template<>
struct PolylineTraits<Vector2f>
{
    using Polyline = MR::Polyline2;
};

template<>
struct PolylineTraits<Vector3f>
{
    using Polyline = MR::Polyline3;
};

/// bounding volume hierarchy for line segments
template<typename V>
class AABBTreePolyline
{
public:
    using Traits = LineTreeTraits<V>;
    using Node = AABBTreeNode<Traits>;
    using NodeId = typename Node::NodeId;

    using NodeVec = Vector<Node, NodeId>;
    [[nodiscard]] const NodeVec& nodes() const
    {
        return nodes_;
    }
    [[nodiscard]] const Node& operator[]( NodeId nid ) const
    {
        return nodes_[nid];
    }
    [[nodiscard]] static NodeId rootNodeId()
    {
        return NodeId{ 0 };
    }
    /// returns the root node bounding box
    [[nodiscard]] Box<V> getBoundingBox() const
    {
        return nodes_.empty() ? Box<V>{} : nodes_[rootNodeId()].box;
    }

    /// creates tree for given polyline
    MRMESH_API AABBTreePolyline( const typename PolylineTraits<V>::Polyline & polyline );
    /// creates tree for selected edges on the mesh (only for 3d tree)
    MRMESH_API AABBTreePolyline( const Mesh& mesh, const UndirectedEdgeBitSet & edgeSet );

    AABBTreePolyline( AABBTreePolyline && ) noexcept = default;
    AABBTreePolyline & operator =( AABBTreePolyline && ) noexcept = default;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return nodes_.heapBytes(); }

private:
    AABBTreePolyline() = default;
    AABBTreePolyline( const AABBTreePolyline & ) = default;
    AABBTreePolyline & operator =( const AABBTreePolyline & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreePolyline>;

    NodeVec nodes_;
};

/// @}

} // namespace MR
