#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRMeshLoadSettings.h"
#include "MROnInit.h"

#include <filesystem>
#include <map>

#define MR_FORMAT_REGISTRY_DECL( ProcName )                                          \
MRMESH_API ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter );            \
MRMESH_API ProcName MR_CONCAT( get, ProcName )( const std::string& extension );      \
MRMESH_API void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName proc ); \
MRMESH_API const IOFilters& getFilters();

#define MR_FORMAT_REGISTRY_IMPL( ProcName )                                   \
ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter )                 \
{                                                                             \
    return FormatRegistry<ProcName>::getProcessor( filter );                  \
}                                                                             \
ProcName MR_CONCAT( get, ProcName )( const std::string& extension )           \
{                                                                             \
    return FormatRegistry<ProcName>::getProcessor( extension );               \
}                                                                             \
void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName processor ) \
{                                                                             \
    FormatRegistry<ProcName>::setProcessor( filter, processor );              \
}                                                                             \
const IOFilters& getFilters()                                                 \
{                                                                             \
    return FormatRegistry<ProcName>::getFilters();                            \
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
    static const IOFilters& getFilters()
    {
        return get_().filters_;
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
    static void setProcessor( const IOFilter& filter, Processor processor )
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
            get_().filters_.emplace_back( filter );
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
    IOFilters filters_;
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
#define MR_ADD_MESH_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::MeshLoad; setMeshLoader( filter, { static_cast<MeshFileLoader>( loader ), static_cast<MeshStreamLoader>( loader ) } ); };

/// \}

}

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

} // namespace MR
