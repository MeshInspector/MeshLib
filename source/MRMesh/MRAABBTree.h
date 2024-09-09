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
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit)
    MRMESH_API void refit( const Mesh & mesh, const VertBitSet & changedVerts );

private:
    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
};

// Not 100% sure what's going on here, `extern template` behind an `#if` is the only option that has worked for me.
// If you make this unconditional, you get undefined references in the python bindings, complaining about the functions defined inline in `MRAABBTreeBase.h` (sic!).
// In theory you just need to instantiate them in the .cpp file, but I couldn't figure out how to do that while preserving their dllexport-ness.
#if !MR_COMPILING_PB11_BINDINGS
extern template class MRMESH_CLASS AABBTreeBase<FaceTreeTraits3>;
#endif

} // namespace MR
