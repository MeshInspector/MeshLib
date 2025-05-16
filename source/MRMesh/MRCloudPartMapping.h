#pragma once

#include "MRMesh/MRMeshFwd.h"

namespace MR
{

// mapping among elements of source point cloud, from which a part is taken, and target (this) point cloud
struct CloudPartMapping
{
    // from.id -> this.id, efficient when full cloud without many invalid points is added into another cloud
    VertMap * src2tgtVerts = nullptr;
    // this.id -> from.id, efficient when any cloud or its part is added into empty cloud
    VertMap * tgt2srcVerts = nullptr;
};

} //namespace MR
