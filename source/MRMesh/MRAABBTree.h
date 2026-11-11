#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector.h"
#include "MRMeshPart.h"

namespace MR
{

/**
 * \defgroup AABBTreeGroup AABB Tree overview
 * \brief This chapter represents documentation about AABB Tree
 */

/// \addtogroup AABBTreeGroup
/// \{

/// node of AABBTree (outside of the class to be default-constructible in its members)
struct AABBTreeWideNode
{
    Box3f box; ///< bounding box of whole subtree
    int first = 0; ///< the first child node, or for a leaf, the first of its triangles in leafTris()
    int count = 0; ///< the number of children, or for a leaf, minus the number of its triangles
    [[nodiscard]] bool leaf() const { return count < 0; }
    [[nodiscard]] int numLeafTris() const { return -count; }
};

/// triangle in a leaf of AABBTree
struct AABBTreeWideLeafTri
{
    FaceId face;
    ThreeVertIds verts; ///< in the order of mesh.topology.getTriVerts( face )
};

/// bounding volume hierarchy of mesh triangles with up to 4 children per node and up to 4 triangles per leaf;
/// the children of a node are stored contiguously, and every leaf keeps the vertices and the boxes of its triangles
class AABBTree
{
public:
    static constexpr int MaxChildren = 4;
    static constexpr int MaxLeafTris = 4;
    /// the deepest level of a node (the root is at 0): the stacks of the traversals are sized for it, and addSplitFaces rebuilds a deeper tree
    static constexpr int MaxDepth = 16;

    using Node = AABBTreeWideNode;
    using LeafTri = AABBTreeWideLeafTri;

    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTree( const MeshPart & mp );

    AABBTree() = default;
    AABBTree( AABBTree && ) noexcept = default;
    AABBTree & operator =( AABBTree && ) noexcept = default;

    /// the root is the first node, if any
    [[nodiscard]] const Vector<Node, NodeId> & nodes() const { return nodes_; }

    /// returns the bounding box of the whole tree (empty for an empty tree)
    [[nodiscard]] Box3f getBoundingBox() const { return nodes_.empty() ? Box3f{} : nodes_.front().box; }

    /// triangles of all leaves, the triangles of each leaf are contiguous
    [[nodiscard]] const Vector<LeafTri> & leafTris() const { return leafTris_; }

    /// the bounding boxes of the triangles in the order of leafTris(), slightly expanded for robust searches
    [[nodiscard]] const Vector<Box3f> & leafBoxes() const { return leafBoxes_; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return nodes_.heapBytes() + leafTris_.heapBytes() + leafBoxes_.heapBytes(); }

    /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;

    /// returns all triangles in the subtree with given root
    [[nodiscard]] MRMESH_API FaceBitSet getSubtreeFaces( NodeId subtreeRoot ) const;

    /// returns the nodes having at least one of given triangles in their subtrees
    [[nodiscard]] MRMESH_API NodeBitSet getNodesFromFaces( const FaceBitSet & faces ) const;

    /// fills map: FaceId -> its position in leafTris();
    /// buffer in faceMap must be resized before the call, and caller is responsible for filling missing face elements
    MRMESH_API void getLeafOrder( FaceBMap & faceMap ) const;

    /// renumbers the triangles and their vertices after the mesh was packed with given maps from old to new ids
    MRMESH_API void renumber( const FaceBMap & faceMap, const VertBMap & vertMap );

    /// changes the order of the vertices of the triangles of given faces (all if nullptr) as MeshTopology::flipOrientation() does;
    /// shall be called after MeshTopology::flipOrientation() if this tree is kept
    MRMESH_API void flipOrientation( const FaceBitSet * region = nullptr );

    /// updates the boxes of the triangles with changed vertices and of their ancestors;
    /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
    /// \param mesh same mesh for which this tree was constructed but with updated coordinates;
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit), nullptr means all vertices
    MRMESH_API void refit( const Mesh & mesh, const VertBitSet * changedVerts );

    /// adds to the tree the faces that appeared after splitting some of its faces into parts (e.g. by Mesh::splitFace or Mesh::splitEdge):
    /// the leaf of each split face gets a subtree with its old triangles and the new parts;
    /// this is a faster alternative to full tree rebuild (but the tree after the update might be less efficient)
    /// \param mesh same mesh for which this tree was constructed, but after the splits
    /// \param new2Old maps every new face (not present in the tree) to the face of this tree it was split from;
    /// it must contain only the splits made since the tree was built or last updated
    MRMESH_API void addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old );

private:
    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
    friend class SharedThreadSafeOwner<AABBTree>;

    Vector<Node, NodeId> nodes_;
    Vector<LeafTri> leafTris_;
    Vector<Box3f> leafBoxes_;
};

/// \}

} // namespace MR
