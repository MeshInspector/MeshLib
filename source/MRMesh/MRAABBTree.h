#pragma once

#include "MRAABBTreeBase.h"
#include "MRVector.h"

namespace MR
{

/**
 * \defgroup AABBTreeGroup AABB Tree overview
 * \brief This chapter represents documentation about AABB Tree
 */

/// bounding volume hierarchy of mesh triangles;
/// the leaves store the vertices of their triangles, and the boxes of nodes and of leaves are kept in one array boxes(),
/// so a change of mesh topology requires either Mesh::invalidateCaches() or updateLeafVerts(),
/// and MeshTopology::flipOrientation() requires flipOrientation() here (as Mesh::flipOrientation() does)
/// \ingroup AABBTreeGroup
class AABBTree : public AABBTreeBase<FaceTreeTraits3>
{
public:
    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTree( const MeshPart & mp );

    AABBTree() = default;
    AABBTree( AABBTree && b ) noexcept : AABBTreeBase( std::move( b ) ), boxes_( std::move( b.boxes_ ) ) { updateBoxBase_(); b.boxBase_ = nullptr; }
    AABBTree & operator =( AABBTree && b ) noexcept
        { AABBTreeBase::operator =( std::move( b ) ); boxes_ = std::move( b.boxes_ ); updateBoxBase_(); b.boxBase_ = nullptr; return *this; }

    /// returns the vertices of the triangle of given leaf
    [[nodiscard]] const ThreeVertIds & leafVerts( NodeId n ) const { return leaf( n ).verts; }

    /// returns the coordinates of the vertices of the triangle of given leaf in the order of leafVerts( n )
    [[nodiscard]] Triangle3f leafTriPoints( const VertCoords & points, NodeId n ) const
        { const auto & v = leafVerts( n ); return { points[v[0]], points[v[1]], points[v[2]] }; }

    /// the bounding boxes of all leaves (the triangles) in the reverse order of leaves() followed by the boxes of all nodes,
    /// so that the box of a node or leaf n is boxes()[numLeaves() + n] (a leaf is encoded as negative n = ~leafIndex)
    using Boxes = Vector<Box3f>;
    [[nodiscard]] const Boxes & boxes() const { return boxes_; }

    /// returns the bounding box of given node or leaf
    [[nodiscard]] const Box3f & box( NodeId n ) const
        { assert( int( n ) >= -int( leaves_.size() ) && int( n ) < int( nodes_.size() ) ); return boxBase_[int( n )]; }

    /// returns the bounding box of the triangle of given leaf
    [[nodiscard]] const Box3f & leafBox( NodeId n ) const { assert( isLeaf( n ) ); return box( n ); }

    /// returns the bounding box of the whole tree
    [[nodiscard]] Box3f getBoundingBox() const { return empty() ? Box3f{} : box( rootNodeId() ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return AABBTreeBase::heapBytes() + boxes_.heapBytes(); }

    /// updates bounding boxes of the nodes containing changed vertices;
    /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
    /// \param mesh same mesh for which this tree was constructed but with updated coordinates;
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit), nullptr means all vertices
    MRMESH_API void refit( const Mesh & mesh, const VertBitSet * changedVerts );

    /// adds to the tree the faces that appeared after splitting some of its faces into parts (e.g. by Mesh::splitFace or Mesh::splitEdge):
    /// the leaf of each split face is replaced with a subtree of that face and its new parts;
    /// this is a faster alternative to full tree rebuild (but the tree after the update might be less efficient)
    /// \param mesh same mesh for which this tree was constructed, but after the splits
    /// \param new2Old maps every new face (not present in the tree) to the face of this tree it was split from;
    /// it must contain only the splits made since the tree was built or last updated: a map accumulated over several calls
    /// would add the earlier faces to the tree once more
    MRMESH_API void addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old );

    /// takes the vertices of all leaves from given topology, e.g. after it was packed
    MRMESH_API void updateLeafVerts( const MeshTopology & topology );

    /// changes the order of the vertices of the triangles of given faces (all if nullptr) as MeshTopology::flipOrientation() does;
    /// shall be called after MeshTopology::flipOrientation() if this tree is kept
    MRMESH_API void flipOrientation( const FaceBitSet * region = nullptr );

private:
    Boxes boxes_;
    Box3f * boxBase_ = nullptr; ///< boxes_.data() + numLeaves(): the box of node or leaf n is boxBase_[n]
    void updateBoxBase_() { boxBase_ = boxes_.data() + leaves_.size(); }
    Box3f & box_( NodeId n ) { return boxBase_[int( n )]; }

    AABBTree( const AABBTree & b ) : AABBTreeBase( b ), boxes_( b.boxes_ ) { updateBoxBase_(); }
    AABBTree & operator =( const AABBTree & b ) { AABBTreeBase::operator =( b ); boxes_ = b.boxes_; updateBoxBase_(); return *this; }
    friend class UniqueThreadSafeOwner<AABBTree>;
    friend class SharedThreadSafeOwner<AABBTree>;
};

} // namespace MR
