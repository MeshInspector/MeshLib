#pragma once

#include "exports.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include <filesystem>
#include <cstring>

namespace MR
{
class Object;

/**
 * \brief save visual object (mesh, lines, points or voxels) to file
 * \param callback - callback function to set progress (for progress bar)
 * \return empty string if no error or error text
 */
MRVIEWER_API VoidOrErrStr saveObjectToFile( const Object& obj, const std::filesystem::path& filename,
                                                               ProgressCallback callback = {} );




}
