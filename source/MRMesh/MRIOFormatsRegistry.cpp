#include "MRIOFormatsRegistry.h"

namespace
{

using namespace MR;

// format processor registry
template <typename T>
class FormatRegistry
{
public:
    using Processor = T;

    // get all registered filters
    static IOFilters getFilters()
    {
        const auto& processors = get_().processors_;
        IOFilters res;
        res.reserve( processors.size() );
        for ( const auto& processor : processors )
            res.emplace_back( processor.filter );
        return res;
    }

    // get a registered processor for the filter
    static Processor getProcessor( IOFilter filter )
    {
        const auto& processors = get_().processors_;
        const auto it = std::find_if( processors.begin(), processors.end(), [&filter] ( auto&& processor )
        {
            return processor.filter.name == filter.name;
        } );
        if ( it == processors.end() )
            return {};
        return it->processor;
    }

    // get a registered processor for the extension
    static Processor getProcessor( const std::string& extension )
    {
        const auto& processors = get_().processors_;
        const auto it = std::find_if( processors.begin(), processors.end(), [&extension] ( auto&& processor )
        {
            return processor.filter.extensions.find( extension ) != std::string::npos;
        } );
        if ( it == processors.end() )
            return {};
        return it->processor;
    }

    // register or update a processor for the filter
    static void setProcessor( IOFilter filter, Processor processor )
    {
        auto& processors = get_().processors_;
        auto it = std::find_if( processors.begin(), processors.end(), [filter] ( auto&& processor )
        {
            return processor.filter.name == filter.name;
        } );
        if ( it != processors.end() )
            it->processor = processor;
        else
            processors.emplace_back( NamedProcessor {filter, processor } );
    }

private:
    FormatRegistry() = default;
    ~FormatRegistry() = default;

    static FormatRegistry<T>& get_()
    {
        static FormatRegistry<T> instance;
        return instance;
    }

    struct NamedProcessor
    {
        IOFilter filter;
        Processor processor;
    };
    std::vector<NamedProcessor> processors_;
};

}

namespace MR
{

const IOFilter AllFilter = { "All (*.*)", "*.*" };

namespace MeshLoad
{

MeshLoaderAdder::MeshLoaderAdder( const NamedMeshLoader& loader )
{
    FormatRegistry<MeshLoader>::setProcessor( loader.filter, loader.loader );
    FormatRegistry<MeshStreamLoader>::setProcessor( loader.filter, loader.streamLoader );
}

MeshLoader getMeshLoader( IOFilter filter )
{
    return FormatRegistry<MeshLoader>::getProcessor( std::move( filter ) );
}

MeshStreamLoader getMeshStreamLoader( IOFilter filter )
{
    return FormatRegistry<MeshStreamLoader>::getProcessor( std::move( filter ) );
}

IOFilters getFilters()
{
    return IOFilters { AllFilter } | FormatRegistry<MeshLoader>::getFilters() | FormatRegistry<MeshStreamLoader>::getFilters();
}

void setMeshLoader( IOFilter filter, MeshLoader loader )
{
    FormatRegistry<MeshLoader>::setProcessor( std::move( filter ), loader );
}

void setMeshStreamLoader( IOFilter filter, MeshStreamLoader streamLoader )
{
    FormatRegistry<MeshStreamLoader>::setProcessor( std::move( filter ), streamLoader );
}

} // namespace MeshLoad

namespace ObjectLoad
{

ObjectLoader getObjectLoader( IOFilter filter )
{
    return FormatRegistry<ObjectLoader>::getProcessor( std::move( filter ) );
}

void setObjectLoader( IOFilter filter, ObjectLoader loader )
{
    FormatRegistry<ObjectLoader>::setProcessor( std::move( filter ), loader );
}

IOFilters getFilters()
{
    // these filters are not used in file dialogs, no need to prepend AllFilter here
    return FormatRegistry<ObjectLoader>::getFilters();
}

ObjectLoaderAdder::ObjectLoaderAdder( IOFilter filter, ObjectLoader loader )
{
    FormatRegistry<ObjectLoader>::setProcessor( std::move( filter ), loader );
}

} // namespace ObjectLoad

namespace AsyncObjectLoad
{

AsyncObjectLoader getObjectLoader( IOFilter filter )
{
    return FormatRegistry<AsyncObjectLoader>::getProcessor( std::move( filter ) );
}

void setObjectLoader( IOFilter filter, AsyncObjectLoader loader )
{
    FormatRegistry<AsyncObjectLoader>::setProcessor( std::move( filter ), loader );
}

IOFilters getFilters()
{
    // these filters are not used in file dialogs, no need to prepend AllFilter here
    return FormatRegistry<AsyncObjectLoader>::getFilters();
}

} // namespace AsyncObjectLoad

namespace ObjectSave
{

ObjectSaver getObjectSaver( IOFilter filter )
{
    return FormatRegistry<ObjectSaver>::getProcessor( std::move( filter ) );
}

ObjectSaver getObjectSaver( const std::string& extension )
{
    return FormatRegistry<ObjectSaver>::getProcessor( extension );
}

void setObjectSaver( IOFilter filter, ObjectSaver saver )
{
    FormatRegistry<ObjectSaver>::setProcessor( std::move( filter ), saver );
}

IOFilters getFilters()
{
    return FormatRegistry<ObjectSaver>::getFilters();
}

ObjectSaverAdder::ObjectSaverAdder( IOFilter filter, ObjectSaver saver )
{
    FormatRegistry<ObjectSaver>::setProcessor( std::move( filter ), saver );
}

} // namespace ObjectSave

} // namespace MR
