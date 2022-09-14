#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <memory>

namespace MR
{

/// removes deleted edges from edge selection and from creases and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void excludeLoneEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh );

/// removes all edges from edge selection and from creases and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void excludeAllEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh );

/// maps edge selection and creases and adds history records for that;
/// it is a good idea to make SCOPED_HISTORY before calling this function
MRVIEWER_API void mapEdgesWithHistory( const std::shared_ptr<ObjectMesh>& objMesh, const EdgeMap & emap );

} //namespace MR
