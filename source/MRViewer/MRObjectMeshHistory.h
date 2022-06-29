#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <memory>

namespace MR
{

/// removes deleted edges from edge selection and from creases and adds history records for that
MRVIEWER_API void clearObjectMeshWithHistory( const std::shared_ptr<ObjectMesh>& objMesh );

} //namespace MR
