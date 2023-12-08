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

using MeshLoader = Expected<MR::Mesh, std::string>( * )( const std::filesystem::path&, const MeshLoadSettings& );
using MeshStreamLoader = Expected<MR::Mesh, std::string>( * )( std::istream&, const MeshLoadSettings& );

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
 * \details loader function signature: Expected<Mesh, std::string> fromFormat( const std::filesystem::path& path, std::vector<Color>* colors );
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

}

}
