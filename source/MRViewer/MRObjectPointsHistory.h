#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <memory>

namespace MR
{

/// packs ObjectPoints updating colors and adds history records for that;
/// \returns false if the cloud was packed before the call and nothing has been changed (and no history records added);
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API bool packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints );

/// sets new valid vertices then packs ObjectPoints updating colors and adds history records for that;
/// \returns false if the cloud was packed before the call and nothing has been changed (and no history records added);
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API bool packPointsWithHistory( const std::shared_ptr<ObjectPoints>& objPoints, VertBitSet newValidVerts );

} //namespace MR
