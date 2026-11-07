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
/// the leaves store the vertices of their triangles instead of the boxes,
/// so a change of mesh topology requires either Mesh::invalidateCaches() or updateLeafVerts()
/// \ingroup AABBTreeGroup
class AABBTree : public AABBTreeBase<FaceTreeTraits3>
{
public:
    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTree( const MeshPart & mp );

    AABBTree() = default;
    AABBTree( AABBTree && ) noexcept = default;
    AABBTree & operator =( AABBTree && ) noexcept = default;

    /// returns the vertices of the triangle of given leaf
    [[nodiscard]] const ThreeVertIds & leafVerts( NodeId n ) const { return leaf( n ).verts; }

    /// computes the bounding box of the triangle of given leaf, insignificantly expanded as all boxes in the tree
    [[nodiscard]] Box3f leafBox( NodeId n, const VertCoords & points ) const
    {
        const auto & v = leafVerts( n );
        Box3f res( points[v[0]], points[v[0]] );
        res.include( points[v[1]] );
        res.include( points[v[2]] );
        return res.insignificantlyExpanded();
    }

    /// returns the bounding box of given node, or computes it for a leaf
    [[nodiscard]] Box3f box( NodeId n, const VertCoords & points ) const { return isLeaf( n ) ? leafBox( n, points ) : nodes_[n].box; }

    /// converts barycentric coordinates relative to leafVerts( n ) into the point on the mesh with given topology,
    /// correct even if the orientation of the triangle was changed after the tree was built
    [[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( const MeshTopology & topology, NodeId n, const TriPointf & bary ) const;

    /// computes the boxes of all leaves in parallel (optionally transformed), indexed as leaves()
    [[nodiscard]] MRMESH_API std::vector<Box3f> computeLeafBoxes( const VertCoords & points, const AffineXf3f * xf = nullptr ) const;

    /// returns the bounding box of the whole tree
    [[nodiscard]] Box3f getBoundingBox() const { return nodes_.empty() ? singleLeafBox_ : nodes_[NodeId( 0 )].box; }

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

    /// takes the vertices of all leaves from given topology, e.g. after it was packed or its orientation was flipped
    MRMESH_API void updateLeafVerts( const MeshTopology & topology );

private:
    Box3f singleLeafBox_; ///< the box of the only leaf if the tree has no nodes

    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
    friend class SharedThreadSafeOwner<AABBTree>;
};

} // namespace MR
