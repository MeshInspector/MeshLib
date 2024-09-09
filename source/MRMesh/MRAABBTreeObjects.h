#pragma once

#include "MRAABBTreeBase.h"
#include "MRMeshOrPoints.h"

namespace MR
{

struct MRMESH_CLASS ObjTreeTraits
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

} // namespace MR
