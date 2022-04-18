#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector.h"
#include "MRVector3.h"

namespace MR
{

// bounding volume hierarchy for point cloud structure
class AABBTreePoints
{
public:
    class NodeTag;
    using NodeId = Id<NodeTag>;

    struct Node
    {
        Box3f box; // bounding box of whole subtree
        NodeId leftOrFirst, rightOrLast; // child nodes indices if >=0, points indices if < 0
        // returns true if node represent real points, false if it has child nodes
        bool leaf() const { return !leftOrFirst.valid(); }
        // returns [first,last) indices of leaf points
        std::pair<int, int> getLeafPointRange() const { assert( leaf() ); return {-( leftOrFirst + 1 ),-( rightOrLast + 1 )}; }
        // sets [first,last) to this node (leaf)
        void setLeafPointRange( int first, int last ) { leftOrFirst = NodeId( -( first + 1 ) ); rightOrLast = NodeId( -( last + 1 ) ); }
    };
    using NodeVec = Vector<Node, NodeId>;
    [[nodiscard]] const NodeVec& nodes() const { return nodes_; }
    [[nodiscard]] const Node& operator[]( NodeId nid ) const { return nodes_[nid]; }
    [[nodiscard]] static NodeId rootNodeId() { return NodeId{0}; }
    // returns the root node bounding box
    [[nodiscard]] Box3f getBoundingBox() const { return nodes_.empty() ? Box3f{} : nodes_[rootNodeId()].box; }

    struct Point
    {
        Vector3f coord;
        VertId id;
    };
    [[nodiscard]] const std::vector<Point>& orderedPoints() const { return orderedPoints_; }

    // creates tree for given point cloud
    MRMESH_API AABBTreePoints( const PointCloud& pointCloud );

    // maximum number of points in leaf node of tree (all of leafs should have this number of points except last one)
    constexpr static int MaxNumPointsInLeaf = 16;

    AABBTreePoints( AABBTreePoints && ) noexcept = default;
    AABBTreePoints & operator =( AABBTreePoints && ) noexcept = default;

    // returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    std::vector<Point> orderedPoints_;
    NodeVec nodes_;

    AABBTreePoints( const AABBTreePoints & ) = default;
    AABBTreePoints & operator =( const AABBTreePoints & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreePoints>;
};

} //namespace MR
