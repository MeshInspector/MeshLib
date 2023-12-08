#include "MRIOFormatsRegistry.h"

namespace
{

using namespace MR;

// format loader registry
template <typename T>
class FormatRegistry
{
public:
    using Loader = T;

    // get all registered filters
    static IOFilters getFilters()
    {
        const auto& loaders = get_().loaders_;
        IOFilters res;
        res.reserve( loaders.size() );
        for ( const auto& loader : loaders )
            res.emplace_back( loader.filter );
        return res;
    }

    // get a registered loader for the filter
    static Loader getLoader( IOFilter filter )
    {
        const auto& loaders = get_().loaders_;
        const auto it = std::find_if( loaders.begin(), loaders.end(), [&filter] ( auto&& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it == loaders.end() )
            return {};
        return it->loader;
    }

    // register or update a loader for the filter
    static void setLoader( IOFilter filter, Loader loader )
    {
        auto& loaders = get_().loaders_;
        auto it = std::find_if( loaders.begin(), loaders.end(), [filter] ( auto&& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it != loaders.end() )
            it->loader = loader;
        else
            loaders.emplace_back( NamedLoader { filter, loader } );
    }

private:
    FormatRegistry() = default;
    ~FormatRegistry() = default;

    static FormatRegistry<T>& get_()
    {
        static FormatRegistry<T> instance;
        return instance;
    }

    struct NamedLoader
    {
        IOFilter filter;
        Loader loader;
    };
    std::vector<NamedLoader> loaders_;
};

}

namespace MR
{

const IOFilter AllFilter = { "All (*.*)", "*.*" };

namespace MeshLoad
{

MeshLoaderAdder::MeshLoaderAdder( const NamedMeshLoader& loader )
{
    FormatRegistry<MeshLoader>::setLoader( loader.filter, loader.loader );
    FormatRegistry<MeshStreamLoader>::setLoader( loader.filter, loader.streamLoader );
}

MeshLoader getMeshLoader( IOFilter filter )
{
    return FormatRegistry<MeshLoader>::getLoader( std::move( filter ) );
}

MeshStreamLoader getMeshStreamLoader( IOFilter filter )
{
    return FormatRegistry<MeshStreamLoader>::getLoader( std::move( filter ) );
}

IOFilters getFilters()
{
    return IOFilters { AllFilter } | FormatRegistry<MeshLoader>::getFilters() | FormatRegistry<MeshStreamLoader>::getFilters();
}

void setMeshLoader( IOFilter filter, MeshLoader loader )
{
    FormatRegistry<MeshLoader>::setLoader( std::move( filter ), loader );
}

void setMeshStreamLoader( IOFilter filter, MeshStreamLoader streamLoader )
{
    FormatRegistry<MeshStreamLoader>::setLoader( std::move( filter ), streamLoader );
}

} // namespace MeshLoad

namespace ObjectLoad
{

ObjectLoader getObjectLoader( IOFilter filter )
{
    return FormatRegistry<ObjectLoader>::getLoader( std::move( filter ) );
}

void setObjectLoader( IOFilter filter, ObjectLoader loader )
{
    FormatRegistry<ObjectLoader>::setLoader( std::move( filter ), loader );
}

IOFilters getFilters()
{
    // these filters are not used in file dialogs, no need to prepend AllFilter here
    return FormatRegistry<ObjectLoader>::getFilters();
}

} // namespace ObjectLoad

namespace AsyncObjectLoad
{

AsyncObjectLoader getObjectLoader( IOFilter filter )
{
    return FormatRegistry<AsyncObjectLoader>::getLoader( std::move( filter ) );
}

void setObjectLoader( IOFilter filter, AsyncObjectLoader loader )
{
    FormatRegistry<AsyncObjectLoader>::setLoader( std::move( filter ), loader );
}

IOFilters getFilters()
{
    // these filters are not used in file dialogs, no need to prepend AllFilter here
    return FormatRegistry<AsyncObjectLoader>::getFilters();
}

} // namespace AsyncObjectLoad

} // namespace MR
