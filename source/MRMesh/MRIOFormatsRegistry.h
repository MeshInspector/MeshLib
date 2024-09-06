#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRMeshLoadSettings.h"
#include "MROnInit.h"
#include "MRPointsLoadSettings.h"
#include "MRSaveSettings.h"

#include <filesystem>
#include <map>

#define MR_FORMAT_REGISTRY_DECL( ProcName )                                                                    \
MRMESH_API ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter );                                      \
MRMESH_API ProcName MR_CONCAT( get, ProcName )( const std::string& extension );                                \
MRMESH_API void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName proc, int8_t priorityScore = 0 ); \
MRMESH_API IOFilters getFilters();

#define MR_FORMAT_REGISTRY_IMPL( ProcName )                                                         \
ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter )                                       \
{                                                                                                   \
    return FormatRegistry<ProcName>::getProcessor( filter );                                        \
}                                                                                                   \
ProcName MR_CONCAT( get, ProcName )( const std::string& extension )                                 \
{                                                                                                   \
    return FormatRegistry<ProcName>::getProcessor( extension );                                     \
}                                                                                                   \
void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName processor, int8_t priorityScore ) \
{                                                                                                   \
    FormatRegistry<ProcName>::setProcessor( filter, processor, priorityScore );                     \
}                                                                                                   \
IOFilters getFilters()                                                                              \
{                                                                                                   \
    return FormatRegistry<ProcName>::getFilters();                                                  \
}

namespace MR
{

MRMESH_API extern const IOFilters AllFilter;

/// format loader registry
/// NOTE: this is a singleton class, do NOT access it from header files
/// you might use MR_FORMAT_REGISTRY_DECL and MR_FORMAT_REGISTRY_IMPL macros to simplify the usage for most common cases
template <typename Processor>
class FormatRegistry
{
public:
    // get all registered filters
    static IOFilters getFilters()
    {
        const auto& filters = get_().filterPriorityQueue_;
        IOFilters results;
        results.reserve( filters.size() );
        for ( const auto& [_, filter] : filters )
            results.emplace_back( filter );
        return results;
    }

    // get a registered loader for the filter
    static Processor getProcessor( const IOFilter& filter )
    {
        const auto& processors = get_().processors_;
        auto it = processors.find( filter );
        if ( it != processors.end() )
            return it->second;
        else
            return {};
    }

    // get a registered loader for the extension
    static Processor getProcessor( const std::string& extension )
    {
        const auto& processors = get_().processors_;
        // TODO: extension cache
        auto it = std::find_if( processors.begin(), processors.end(), [&extension] ( auto&& item )
        {
            const auto& [filter, _] = item;
            return filter.extensions.find( extension ) != std::string::npos;
        } );
        if ( it != processors.end() )
            return it->second;
        else
            return {};
    }

    // register or update a loader for the filter
    static void setProcessor( const IOFilter& filter, Processor processor, int8_t priorityScore = 0 )
    {
        auto& processors = get_().processors_;
        auto it = processors.find( filter );
        if ( it != processors.end() )
        {
            it->second = processor;
        }
        else
        {
            processors.emplace( filter, processor );

            auto& filters = get_().filterPriorityQueue_;
            filters.emplace( priorityScore, filter );
        }
    }

private:
    FormatRegistry() = default;
    ~FormatRegistry() = default;

    static FormatRegistry<Processor>& get_()
    {
        static FormatRegistry<Processor> instance;
        return instance;
    }

    std::map<IOFilter, Processor> processors_;
    std::multimap<int8_t, IOFilter> filterPriorityQueue_;
};

namespace MeshLoad
{

/// \defgroup IOFormatsRegistryGroup IO Formats Registry
/// \ingroup IOGroup
/// \{

using MeshFileLoader = Expected<MR::Mesh>( * )( const std::filesystem::path&, const MeshLoadSettings& );
using MeshStreamLoader = Expected<MR::Mesh>( * )( std::istream&, const MeshLoadSettings& );

struct MeshLoader
{
    MeshFileLoader fileLoad{ nullptr };
    MeshStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( MeshLoader )

/** 
 * \brief Register filter with loader function
 * \details loader function signature: Expected<Mesh> fromFormat( const std::filesystem::path& path, const MeshLoadSettings& settings );
 * example:
 * MR_ADD_MESH_LOADER( IOFilter("Name of filter (.ext)","*.ext"), fromFormat)
 */
#define MR_ADD_MESH_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::MeshLoad; setMeshLoader( filter, { static_cast<MeshFileLoader>( loader ), static_cast<MeshStreamLoader>( loader ) } __VA_OPT__(,)__VA_ARGS__ ); };

/// \}

} // namespace MeshLoad

namespace MeshSave
{

using MeshFileSaver = Expected<void>( * )( const Mesh&, const std::filesystem::path&, const SaveSettings& );
using MeshStreamSaver = Expected<void>( * )( const Mesh&, std::ostream&, const SaveSettings& );

struct MeshSaver
{
    MeshFileSaver fileSave{ nullptr };
    MeshStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( MeshSaver )

#define MR_ADD_MESH_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::MeshSave; setMeshSaver( filter, { static_cast<MeshFileSaver>( saver ), static_cast<MeshStreamSaver>( saver ) } __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace MeshSave

namespace LinesLoad
{

using LinesFileLoader = Expected<Polyline3>( * )( const std::filesystem::path&, ProgressCallback );
using LinesStreamLoader = Expected<Polyline3>( * )( std::istream&, ProgressCallback );

struct LinesLoader
{
    LinesFileLoader fileLoad{ nullptr };
    LinesStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( LinesLoader )

#define MR_ADD_LINES_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::LinesLoad; setLinesLoader( filter, { static_cast<LinesFileLoader>( loader ), static_cast<LinesStreamLoader>( loader ) } __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace LinesLoad

namespace LinesSave
{

using LinesFileSaver = Expected<void>( * )( const Polyline3&, const std::filesystem::path&, const SaveSettings& );
using LinesStreamSaver = Expected<void>( * )( const Polyline3&, std::ostream&, const SaveSettings& );

struct LinesSaver
{
    LinesFileSaver fileSave{ nullptr };
    LinesStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( LinesSaver )

#define MR_ADD_LINES_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::LinesSave; setLinesSaver( filter, { static_cast<LinesFileSaver>( saver ), static_cast<LinesStreamSaver>( saver ) } __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace LinesSave

namespace PointsLoad
{

using PointsFileLoader = Expected<PointCloud>( * )( const std::filesystem::path&, const PointsLoadSettings& );
using PointsStreamLoader = Expected<PointCloud>( * )( std::istream&, const PointsLoadSettings& );

struct PointsLoader
{
    PointsFileLoader fileLoad{ nullptr };
    PointsStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( PointsLoader )

#define MR_ADD_POINTS_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::PointsLoad; setPointsLoader( filter, { static_cast<PointsFileLoader>( loader ), static_cast<PointsStreamLoader>( loader ) } __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace PointsLoad

namespace PointsSave
{

using PointsFileSaver = Expected<void>( * )( const PointCloud&, const std::filesystem::path&, const SaveSettings& );
using PointsStreamSaver = Expected<void>( * )( const PointCloud&, std::ostream&, const SaveSettings& );

struct PointsSaver
{
    PointsFileSaver fileSave{ nullptr };
    PointsStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( PointsSaver )

#define MR_ADD_POINTS_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::PointsSave; setPointsSaver( filter, { static_cast<PointsFileSaver>( saver ), static_cast<PointsStreamSaver>( saver ) } __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace PointsSave

namespace ImageLoad
{

using ImageLoader = Expected<Image>( * )( const std::filesystem::path& );

MR_FORMAT_REGISTRY_DECL( ImageLoader )

#define MR_ADD_IMAGE_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::ImageLoad; setImageLoader( filter, loader __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace ImageLoad

namespace ImageSave
{

using ImageSaver = Expected<void>( * )( const Image&, const std::filesystem::path& );

MR_FORMAT_REGISTRY_DECL( ImageSaver )

#define MR_ADD_IMAGE_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::ImageSave; setImageSaver( filter, saver __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace ImageSave

#ifndef MRMESH_NO_OPENVDB
namespace VoxelsLoad
{

using VoxelsLoader = Expected<std::vector<VdbVolume>>( * )( const std::filesystem::path&, const ProgressCallback& );

MR_FORMAT_REGISTRY_DECL( VoxelsLoader )

#define MR_ADD_VOXELS_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::VoxelsLoad; setVoxelsLoader( filter, loader __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace VoxelsLoad

namespace VoxelsSave
{

using VoxelsSaver = Expected<void>( * )( const VdbVolume&, const std::filesystem::path&, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( VoxelsSaver )

#define MR_ADD_VOXELS_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::VoxelsSave; setVoxelsSaver( filter, saver __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace VoxelsSave
#endif

using ObjectPtr = std::shared_ptr<Object>;

namespace ObjectLoad
{

using ObjectLoader = Expected<std::vector<ObjectPtr>>( * )( const std::filesystem::path&, std::string*, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( ObjectLoader )

} // namespace ObjectLoad

namespace AsyncObjectLoad
{

using PostLoadCallback = std::function<void ( Expected<std::vector<ObjectPtr>> )>;
using ObjectLoader = void( * )( const std::filesystem::path&, std::string*, PostLoadCallback, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( ObjectLoader )

} // namespace AsyncObjectLoad

namespace SceneLoad
{

using SceneLoader = Expected<ObjectPtr>( * )( const std::filesystem::path&, std::string*, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( SceneLoader )

#define MR_ADD_SCENE_LOADER( filter, loader, ... ) \
MR_ON_INIT { using namespace MR::SceneLoad; setSceneLoader( filter, loader __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace SceneLoad

namespace SceneSave
{

using SceneSaver = Expected<void>( * )( const Object&, const std::filesystem::path&, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( SceneSaver )

#define MR_ADD_SCENE_SAVER( filter, saver, ... ) \
MR_ON_INIT { using namespace MR::SceneSave; setSceneSaver( filter, saver __VA_OPT__(,)__VA_ARGS__ ); };

} // namespace SceneSave

} // namespace MR
