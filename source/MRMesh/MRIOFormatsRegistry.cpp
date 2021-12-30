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

    static void addLoader( const NamedMeshLoader& loader )
    {
        auto& loaders = get_().loaders_;
        if ( loaders.empty() )
            loaders.push_back( {AllFilter,{}} );
        loaders.push_back( loader );
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

IOFilters getFilters()
{
    return FormatsRegistry::getFilters();
}

}
}
