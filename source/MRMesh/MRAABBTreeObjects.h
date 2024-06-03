#pragma once

#include "MRAABBTreeBase.h"

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
};

} // namespace MR
