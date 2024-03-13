#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include <filesystem>

namespace MR
{

/// \ingroup DataModelGroup
/// \{

/// loads mesh from given file in new object
MRMESH_API Expected<ObjectMesh, std::string> makeObjectMeshFromFile( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from given file and makes either ObjectMesh or ObjectPoints (if the file has points but not faces)
MRMESH_API Expected<std::shared_ptr<Object>, std::string> makeObjectFromMeshFile( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads lines from given file in new object
MRMESH_API Expected<ObjectLines, std::string> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads points from given file in new object
MRMESH_API Expected<ObjectPoints, std::string> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads distance map from given file in new object
MRMESH_API Expected<ObjectDistanceMap, std::string> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads gcode from given file in new object
MRMESH_API Expected<ObjectGcode, std::string> makeObjectGcodeFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
/// loads voxels from given file in new object
MRMESH_API Expected<std::vector<std::shared_ptr<ObjectVoxels>>, std::string> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );
#endif

/**
 * \brief load object (mesh, lines, points, voxels or scene) from file
 * \param loadWarn - string that collect warning messages
 * \param callback - callback function to set progress (for progress bar)
 * \return empty string if no error or error text
 */
MRMESH_API Expected<std::vector<std::shared_ptr<Object>>, std::string> loadObjectFromFile( const std::filesystem::path& filename,
                                                                                           std::string* loadWarn = nullptr, ProgressCallback callback = {} );

// check if there are any supported files folder and subfolders
MRMESH_API bool isSupportedFileInSubfolders( const std::filesystem::path& folder );

/// loads meshes from given folder in new container object
MRMESH_API Expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path& folder, ProgressCallback callback = {} );

//tries to load scene from every format listed in SceneFormatFilters
MRMESH_API Expected<std::shared_ptr<Object>, std::string> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

/// \}

} // namespace MR
