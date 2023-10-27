#include "MRIOFormatsRegistry.h"

namespace MR
{
const IOFilter AllFilter = {"All (*.*)",         "*.*"};

namespace MeshLoad
{

class FormatsRegistry
{
public:

    static IOFilters getFilters()
    {
        const auto& loaders = get_().loaders_;
        IOFilters res( loaders.size() );
        for ( size_t i = 0; i < loaders.size(); ++i )
            res[i] = loaders[i].filter;
        return res;
    }

    static MeshLoader getLoader( IOFilter filter )
    {
        const auto& loaders = get_().loaders_;
        auto it = std::find_if( loaders.begin(), loaders.end(), [filter]( const NamedMeshLoader& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it != loaders.end() )
            return it->loader;
        return {};
    }

    static MeshStreamLoader getStreamLoader( IOFilter filter )
    {
        const auto& loaders = get_().loaders_;
        auto it = std::find_if( loaders.begin(), loaders.end(), [filter] ( const NamedMeshLoader& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it != loaders.end() )
            return it->streamLoader;
        return {};
    }

    static void addLoader( const NamedMeshLoader& loader )
    {
        auto& loaders = get_().loaders_;
        if ( loaders.empty() )
            loaders.push_back( {AllFilter,{}} );
        loaders.push_back( loader );
    }

    static void setLoader( IOFilter filter, MeshLoader loader )
    {
        auto& loaders = get_().loaders_;
        auto it = std::find_if( loaders.begin(), loaders.end(), [filter] ( auto&& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it != loaders.end() )
            it->loader = loader;
        else
            loaders.emplace_back( NamedMeshLoader { filter, loader, nullptr } );
    }

    static void setStreamLoader( IOFilter filter, MeshStreamLoader streamLoader )
    {
        auto& loaders = get_().loaders_;
        auto it = std::find_if( loaders.begin(), loaders.end(), [filter] ( auto&& loader )
        {
            return loader.filter.name == filter.name;
        } );
        if ( it != loaders.end() )
            it->streamLoader = streamLoader;
        else
            loaders.emplace_back( NamedMeshLoader { filter, nullptr, streamLoader } );
    }

private:
    FormatsRegistry() = default;
    ~FormatsRegistry() = default;

    static FormatsRegistry& get_()
    {
        static FormatsRegistry instance;
        return instance;
    }
    std::vector<NamedMeshLoader> loaders_;
};

MeshLoaderAdder::MeshLoaderAdder( const NamedMeshLoader& loader )
{
    FormatsRegistry::addLoader( loader );
}

MeshLoader getMeshLoader( IOFilter filter )
{
    return FormatsRegistry::getLoader( filter );
}

MeshStreamLoader getMeshStreamLoader( IOFilter filter )
{
    return FormatsRegistry::getStreamLoader( filter );
}

IOFilters getFilters()
{
    return FormatsRegistry::getFilters();
}

void setMeshLoader( IOFilter filter, MeshLoader loader )
{
    FormatsRegistry::setLoader( filter, loader );
}

void setMeshStreamLoader( IOFilter filter, MeshStreamLoader streamLoader )
{
    FormatsRegistry::setStreamLoader( filter, streamLoader );
}

}
}
