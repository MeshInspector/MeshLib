#pragma once

#include "MRAABBTreeBase.h"
#include "MRMeshOrPoints.h"

namespace MR
{

struct ObjTreeTraits
{
    using LeafTag = ObjTag;
    using LeafId = ObjId;
    using BoxT = Box3f;
};

/// tree containing world bounding boxes of individual objects having individual local-to-world transformations
/// \ingroup AABBTreeGroup
class AABBTreeObjects : public AABBTreeBase<ObjTreeTraits>
{
public:
    AABBTreeObjects() = default;

    /// creates tree for given set of objects each with its own transformation
    [[nodiscard]] MRMESH_API explicit AABBTreeObjects( Vector<MeshOrPointsXf, ObjId> objs );

    /// gets object by its id
    [[nodiscard]] const MeshOrPoints & obj( ObjId oi ) const { return objs_[oi].obj; }

    /// gets transformation from local space of given object to world space
    [[nodiscard]] const AffineXf3f & toWorld( ObjId oi ) const { return objs_[oi].xf; }

    /// gets transformation from world space to local space of given object
    [[nodiscard]] const AffineXf3f & toLocal( ObjId oi ) const { return toLocal_[oi]; }

    /// gets mapping: objId -> its transformation from world space to local space
    [[nodiscard]] const Vector<AffineXf3f, ObjId> & toLocal() const { return toLocal_; }

private:
    Vector<MeshOrPointsXf, ObjId> objs_;
    Vector<AffineXf3f, ObjId> toLocal_;
};

// Not 100% sure what's going on here, and whether the condition is 100% correct. `extern template` behind an `#if` is the only option that has worked for me.
// If you make this unconditional, you get undefined references in the python bindings, complaining about the functions defined inline in `MRAABBTreeBase.h` (sic!).
// In theory you just need to instantiate them in the .cpp file, but I couldn't figure out how to do that while preserving their dllexport-ness.
#if defined(_MSC_VER) && !defined(__clang__)
extern template class MRMESH_CLASS AABBTreeBase<ObjTreeTraits>;
#endif

} // namespace MR
