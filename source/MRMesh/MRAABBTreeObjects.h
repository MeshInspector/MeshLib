#pragma once

#include "MRAABBTreeBase.h"
#include "MRAffineXf3.h"

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
    [[nodiscard]] MRMESH_API explicit AABBTreeObjects( const Vector<MeshOrPointsXf, ObjId> & objs );

    /// get mapping: objId -> its tranfomation from local space to world space
    [[nodiscard]] const Vector<AffineXf3f, ObjId> & toWorld() const { return toWorld_; }

    /// get mapping: objId -> its tranfomation from world space to local space
    [[nodiscard]] const Vector<AffineXf3f, ObjId> & toLocal() const { return toLocal_; }

private:
    Vector<AffineXf3f, ObjId> toWorld_, toLocal_;
};

} // namespace MR
