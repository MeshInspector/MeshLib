#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <memory>

namespace MR
{

/// removes deleted edges from edge selection and from creases and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void clearObjectMeshWithHistory( const std::shared_ptr<ObjectMesh>& objMesh );

} //namespace MR
