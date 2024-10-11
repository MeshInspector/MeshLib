#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRUniqueTemporaryFolder.h"

#include <filesystem>

namespace MR
{

namespace ObjectSave
{

/// save an object tree to a given file
/// file format must be scene-capable
MRMESH_API Expected<void> toAnySupportedSceneFormat( const Object& object, const std::filesystem::path& file,
                                                     ProgressCallback callback = {} );

/// save a scene object to a given file
/// if the file format is scene-capable, saves all the object's entities
/// otherwise, saves only merged entities of the corresponding type (mesh, polyline, point cloud, etc.)
MRMESH_API Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                                ProgressCallback callback = {} );

} // namespace ObjectSave

/**
 * \brief saves object subtree in given scene file (zip/mru)
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *
 * if preCompress is set, it is called before compression
 * saving is controlled with Object::serializeModel_ and Object::serializeFields_
 */
MRMESH_API Expected<void> serializeObjectTree( const Object& object, const std::filesystem::path& path,
                                             ProgressCallback progress, FolderCallback preCompress );
MRMESH_API Expected<void> serializeObjectTree( const Object& object, const std::filesystem::path& path,
                                             ProgressCallback progress = {} );

} // namespace MR
