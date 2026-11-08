#pragma once

#include "MRAABBTreeBase.h"

namespace MR
{

/// binary bounding volume hierarchy of mesh triangles with one triangle per leaf;
/// only the CUDA algorithms use it until they are ported to AABBTree
/// \ingroup AABBTreeGroup
class AABBTreeBinary : public AABBTreeBase<FaceTreeTraits3>
{
public:
    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTreeBinary( const MeshPart & mp );

    AABBTreeBinary() = default;
    AABBTreeBinary( AABBTreeBinary && ) noexcept = default;
    AABBTreeBinary & operator =( AABBTreeBinary && ) noexcept = default;

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

private:
    AABBTreeBinary( const AABBTreeBinary & ) = default;
    AABBTreeBinary & operator =( const AABBTreeBinary & ) = default;
    friend class UniqueThreadSafeOwner<AABBTreeBinary>;
    friend class SharedThreadSafeOwner<AABBTreeBinary>;
};

} // namespace MR
