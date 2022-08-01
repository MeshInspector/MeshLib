#pragma once
#include "exports.h"
#include "MRMesh/MRProgressCallback.h"
#include <filesystem>
#include <cstring>
#include <tl/expected.hpp>

namespace MR
{
class Object;

/**
 * \brief save visual object (mesh, lines, points or voxels) to file
 * \param callback - callback function to set progress (for progress bar)
 * \return empty string if no error or error text
 */
MRVIEWER_API tl::expected<void, std::string> saveObjectToFile( const Object& obj, const std::filesystem::path& filename,
                                                               ProgressCallback callback = {} );

/**
 * \brief load object (mesh, lines, points, voxels or scene) from file
 * \param callback - callback function to set progress (for progress bar)
 * \return empty string if no error or error text
 */
MRVIEWER_API tl::expected<std::vector<std::shared_ptr<Object>>, std::string> loadObjectFromFile( const std::filesystem::path& filename,
                                                                                                 ProgressCallback callback = {} );



}
