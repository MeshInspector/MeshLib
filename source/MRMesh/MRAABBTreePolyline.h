#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

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
    using Polyline = MR::Polyline;
};

// bounding volume hierarchy for line segments
template<typename V>
class AABBTreePolyline
{
public:
    using Traits = LineTreeTraits<V>;
    using Node = AABBTreeNode<Traits>;
    using NodeId = typename Node::NodeId;

    using NodeVec = Vector<Node, NodeId>;
    const NodeVec& nodes() const
    {
        return nodes_;
    }
    const Node& operator[]( NodeId nid ) const
    {
        return nodes_[nid];
    }
    static NodeId rootNodeId()
    {
        return NodeId{ 0 };
    }
    // returns the root node bounding box
    Box<V> getBoundingBox() const
    {
        return nodes_.empty() ? Box<V>{} : nodes_[rootNodeId()].box;
    }

    // creates tree for given polyline
    MRMESH_API AABBTreePolyline( const typename PolylineTraits<V>::Polyline & polyline );
    // creates tree for selected edges on the mesh (only for 3d tree)
    MRMESH_API AABBTreePolyline( const Mesh& mesh, const UndirectedEdgeBitSet & edgeSet );

    AABBTreePolyline( AABBTreePolyline && ) noexcept = default;
    AABBTreePolyline & operator =( AABBTreePolyline && ) noexcept = default;

private:
    AABBTreePolyline() = default;
    AABBTreePolyline( const AABBTreePolyline & ) = default;
    AABBTreePolyline & operator =( const AABBTreePolyline & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreePolyline>;

    NodeVec nodes_;
};

} //namespace MR
