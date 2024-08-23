#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRMeshLoadSettings.h"
#include "MRProgressCallback.h"

#include <filesystem>

namespace MR
{

namespace MeshLoad
{

/// \defgroup IOFormatsRegistryGroup IO Formats Registry
/// \ingroup IOGroup
/// \{

using MeshLoader = Expected<MR::Mesh>( * )( const std::filesystem::path&, const MeshLoadSettings& );
using MeshStreamLoader = Expected<MR::Mesh>( * )( std::istream&, const MeshLoadSettings& );

struct NamedMeshLoader
{
    IOFilter filter;
    MeshLoader loader{ nullptr };
    MeshStreamLoader streamLoader{ nullptr };
};

/// Finds expected loader from registry
MRMESH_API MeshLoader getMeshLoader( IOFilter filter );
/// Finds expected loader from registry
MRMESH_API MeshStreamLoader getMeshStreamLoader( IOFilter filter );
/// Gets all registered filters
MRMESH_API IOFilters getFilters();

/// Add or override a loader in the registry
MRMESH_API void setMeshLoader( IOFilter filter, MeshLoader loader );
/// Add or override a loader in the registry
MRMESH_API void setMeshStreamLoader( IOFilter filter, MeshStreamLoader streamLoader );

/** 
 * \brief Register filter with loader function
 * \details loader function signature: Expected<Mesh> fromFormat( const std::filesystem::path& path, std::vector<Color>* colors );
 * example:
 * ADD_MESH_LOADER( IOFilter("Name of filter (.ext)","*.ext"), fromFormat)
 */
#define MR_ADD_MESH_LOADER( filter, loader ) \
MR::MeshLoad::MeshLoaderAdder __meshLoaderAdder_##loader(MR::MeshLoad::NamedMeshLoader{filter,static_cast<MR::MeshLoad::MeshLoader>(loader),static_cast<MR::MeshLoad::MeshStreamLoader>(loader)});\

class MeshLoaderAdder
{
public:
    MRMESH_API MeshLoaderAdder( const NamedMeshLoader& loader );
};

/// \}

}

using ObjectPtr = std::shared_ptr<Object>;

namespace ObjectLoad
{

using ObjectLoader = Expected<std::vector<ObjectPtr>>( * )( const std::filesystem::path&, std::string*, ProgressCallback );

/// Find an appropriate loader from the registry
MRMESH_API ObjectLoader getObjectLoader( IOFilter filter );
/// Add or override a loader in the registry
MRMESH_API void setObjectLoader( IOFilter filter, ObjectLoader loader );
/// Get all registered filters
MRMESH_API IOFilters getFilters();

class ObjectLoaderAdder
{
public:
    MRMESH_API ObjectLoaderAdder( IOFilter filter, ObjectLoader loader );
};

/**
 * \brief Register filter with loader function
 * \details loader function signature: Expected<std::vector<std::shared_ptr<Object>>> fromFormat( const std::filesystem::path& path, std::string* warnings, ProgressCallback cb );
 * example:
 * MR_ADD_OBJECT_LOADER( IOFilter( "Name of filter (.ext)", "*.ext" ), fromFormat )
 */
#define MR_ADD_OBJECT_LOADER( filter, loader ) \
MR::ObjectLoad::ObjectLoaderAdder MR_CONCAT( __objectLoaderAdder_, __LINE__ )( filter, loader );

} // namespace ObjectLoad

namespace AsyncObjectLoad
{

using PostLoadCallback = std::function<void ( Expected<std::vector<ObjectPtr>> )>;
using AsyncObjectLoader = void( * )( const std::filesystem::path&, std::string*, PostLoadCallback, ProgressCallback );

/// Find an appropriate loader from the registry
MRMESH_API AsyncObjectLoader getObjectLoader( IOFilter filter );
/// Add or override a loader in the registry
MRMESH_API void setObjectLoader( IOFilter filter, AsyncObjectLoader loader );
/// Get all registered filters
MRMESH_API IOFilters getFilters();

} // namespace AsyncObjectLoad

namespace ObjectSave
{

using ObjectSaver = Expected<void>( * )( const Object&, const std::filesystem::path&, ProgressCallback );

/// Find an appropriate loader from the registry
MRMESH_API ObjectSaver getObjectSaver( IOFilter filter );
MRMESH_API ObjectSaver getObjectSaver( const std::string& extension );
/// Add or override a loader in the registry
MRMESH_API void setObjectSaver( IOFilter filter, ObjectSaver saver );
/// Get all registered filters
MRMESH_API IOFilters getFilters();

class ObjectSaverAdder
{
public:
    MRMESH_API ObjectSaverAdder( IOFilter filter, ObjectSaver saver );
};

/**
 * \brief Register filter with saver function
 * \details saver function signature: Expected<void> toFormat( const Object& object, const std::filesystem::path& path, ProgressCallback cb );
 * example:
 * MR_ADD_OBJECT_SAVER( IOFilter( "Name of filter (.ext)", "*.ext" ), toFormat )
 */
#define MR_ADD_OBJECT_SAVER( filter, saver ) \
MR::ObjectSave::ObjectSaverAdder MR_CONCAT( __objectSaverAdder_, __LINE__ )( filter, saver );

} // namespace ObjectSave

}
