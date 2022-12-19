#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <filesystem>
#include <tl/expected.hpp>

namespace MR
{

/// \ingroup DataModelGroup
/// \{

/// loads mesh from given file in new object
MRMESH_API tl::expected<ObjectMesh, std::string> makeObjectMeshFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads lines from given file in new object
MRMESH_API tl::expected<ObjectLines, std::string> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads points from given file in new object
MRMESH_API tl::expected<ObjectPoints, std::string> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads distance map from given file in new object
MRMESH_API tl::expected<ObjectDistanceMap, std::string> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

// check if there are any supported files folder and subfolders
MRMESH_API bool isAnySupportedFilesInSubfolders( const std::filesystem::path& folder );

/// loads meshes from given folder in new container object
MRMESH_API tl::expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path& folder, ProgressCallback callback = {} );

/// \}

} // namespace MR
