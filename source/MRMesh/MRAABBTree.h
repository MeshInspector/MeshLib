#pragma once

#include "MRAABBTreeNode.h"
#include "MRVector.h"

namespace MR
{

/**
 * \defgroup AABBTreeGroup AABB Tree overview
 * \brief This chapter represents documentation about AABB Tree
 */

/// bounding volume hierarchy
/// \ingroup AABBTreeGroup
class AABBTree
{
public:
    using Node = AABBTreeNode<FaceTreeTraits3>;
    using NodeId = AABBTreeNode<FaceTreeTraits3>::NodeId;
    using NodeBitSet = TaggedBitSet<AABBTreeNode<FaceTreeTraits3>::NodeTag>;

    using NodeVec = Vector<Node, NodeId>;
    [[nodiscard]] const NodeVec & nodes() const { return nodes_; }
    [[nodiscard]] const Node & operator[]( NodeId nid ) const { return nodes_[nid]; }
    [[nodiscard]] static NodeId rootNodeId() { return NodeId{ 0 }; }
    /// returns the root node bounding box
    [[nodiscard]] Box3f getBoundingBox() const { return nodes_.empty() ? Box3f{} : nodes_[rootNodeId()].box; }
    /// returns true if the tree contains exactly the same number of triangles as in given mesh;
    /// this is fast validity check, but it is not comprehensive (tree can be outdated even if true is returned)
    [[nodiscard]] MRMESH_API bool containsSameNumberOfTris( const Mesh & mesh ) const;

    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API AABBTree( const MeshPart & mp );

    /// returns all faces in the subtree with given root
    [[nodiscard]] MRMESH_API FaceBitSet getSubtreeFaces( NodeId subtreeRoot ) const;
    /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;
    /// returns FaceId -> leaf#;
    /// buffer in faceMap must be resized before the call, and caller is responsible for filling missing face elements
    MRMESH_API void getLeafOrder( FaceBMap & faceMap ) const;
    /// returns FaceId -> leaf#, then resets leaf order to 0,1,2,...;
    /// buffer in faceMap must be resized before the call, and caller is responsible for filling missing face elements
    MRMESH_API void getLeafOrderAndReset( FaceBMap & faceMap );

    /// returns set of nodes containing among direct or indirect children given faces
    [[nodiscard]] MRMESH_API NodeBitSet getNodesFromFaces( const FaceBitSet & faces ) const;

    AABBTree( AABBTree && ) noexcept = default;
    AABBTree & operator =( AABBTree && ) noexcept = default;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const { return nodes_.heapBytes(); }

    /// updates bounding boxes of the nodes containing changed vertices;
    /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
    /// \param mesh same mesh for which this tree was constructed but with updated coordinates;
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit)
    MRMESH_API void refit( const Mesh & mesh, const VertBitSet & changedVerts );

private:
    NodeVec nodes_;

    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
};

} // namespace MR
