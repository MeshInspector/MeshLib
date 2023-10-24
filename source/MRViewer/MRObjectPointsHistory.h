#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <memory>

namespace MR
{

/// packs ObjectPoints optimally updating colors and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reoder );

/// sets new valid vertices then packs ObjectPoints optimally updating colors and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, Reorder reoder, VertBitSet newValidVerts );

} //namespace MR
