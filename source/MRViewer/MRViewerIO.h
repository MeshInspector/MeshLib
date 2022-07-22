#pragma once
#include "exports.h"
#include "MRMesh\MRProgressCallback.h"

namespace MR
{
class VisualObject;

MRVIEWER_API std::string saveObjectToFile( const std::shared_ptr<VisualObject>& obj, const std::filesystem::path& filename,
                                           ProgressCallback callback = emptyProgressCallback );
}
