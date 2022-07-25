#pragma once
#include "exports.h"
#include "MRMesh/MRProgressCallback.h"
#include <filesystem>
#include <cstring>

namespace MR
{
class VisualObject;

/**
 * \brief save visual object (mesh, lines, points or voxels) to file
 * \param callback - callback function to set progress (for progress bar)
 */
MRVIEWER_API std::string saveObjectToFile( const std::shared_ptr<VisualObject>& obj, const std::filesystem::path& filename,
                                           ProgressCallback callback = {} );

}
