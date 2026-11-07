#pragma once

#include "MRAABBTreeBase.h"

namespace MR
{

/**
 * \defgroup AABBTreeGroup AABB Tree overview
 * \brief This chapter represents documentation about AABB Tree
 */

/// bounding volume hierarchy
/// \ingroup AABBTreeGroup
class AABBTree : public AABBTreeBase<FaceTreeTraits3>
{
public:
    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTree( const MeshPart & mp );

    AABBTree() = default;
    AABBTree( AABBTree && ) noexcept = default;
    AABBTree & operator =( AABBTree && ) noexcept = default;

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

    /// vertex ids of the faces of all leaves in the order of Node::leafPos(), which is the order of leaf nodes after construction
    [[nodiscard]] const Vector<ThreeVertIds> & leafTriangulation() const { return leafTris_; }

    /// returns vertex ids of the face of given leaf node
    [[nodiscard]] const ThreeVertIds & leafTriVerts( const Node & leaf ) const { return leafTris_[leaf.leafPos()]; }

    /// returns the coordinates of the vertices of the face of given leaf node in the order of leafTriVerts()
    [[nodiscard]] Triangle3f leafTriPoints( const VertCoords & points, const Node & leaf ) const
        { const auto & t = leafTriVerts( leaf ); return { points[t[0]], points[t[1]], points[t[2]] }; }

    /// takes vertex ids of all tree's faces from given topology, e.g. after its vertices were renumbered
    MRMESH_API void updateTriangulation( const MeshTopology & topology );

    /// converts barycentric coordinates relative to vertices (t) of face (f) into the point relative to topology.edgeWithLeft( f ),
    /// which can list the same vertices in another order (e.g. after topology.flipOrientation() with this tree kept)
    [[nodiscard]] MRMESH_API static MeshTriPoint toMeshTriPoint( const MeshTopology & topology, FaceId f, const ThreeVertIds & t, const TriPointf & bary );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return AABBTreeBase::heapBytes() + leafTris_.heapBytes(); }

private:
    Vector<ThreeVertIds> leafTris_;

    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
    friend class SharedThreadSafeOwner<AABBTree>;
};

} // namespace MR
