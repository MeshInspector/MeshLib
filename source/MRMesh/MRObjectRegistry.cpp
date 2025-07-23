#include "MRObjectRegistry.h"

namespace
{

std::unordered_multimap<std::string, std::shared_ptr<void>> cObjectRegistry = {};
std::shared_mutex cObjectRegistryMutex = {};

} // namespace

namespace MR::detail
{

void registerObject( const std::string& typeName, std::shared_ptr<void> objectPtr )
{
    std::unique_lock lock( cObjectRegistryMutex );
    cObjectRegistry.emplace( typeName, std::move( objectPtr ) );
}

void unregisterObject( const std::string& typeName, void* objectPtr )
{
    std::unique_lock lock( cObjectRegistryMutex );
    for ( auto [it, end] = cObjectRegistry.equal_range( std::string { typeName } ); it != end; ++it )
    {
        auto& [_, ptr] = *it;
        if ( ptr.get() == objectPtr )
        {
            cObjectRegistry.erase( it );
            return;
        }
    }
}

std::vector<std::shared_ptr<void>> getObjects( const std::string& typeName )
{
    std::shared_lock lock( cObjectRegistryMutex );
    std::vector<std::shared_ptr<void>> results;
    for ( auto [it, end] = cObjectRegistry.equal_range( typeName ); it != end; ++it )
    {
        auto& [_, ptr] = *it;
        results.emplace_back( ptr );
    }
    return results;
}

std::shared_ptr<void> getObject( const std::string& typeName )
{
    std::shared_lock lock( cObjectRegistryMutex );
    if ( auto it = cObjectRegistry.find( typeName ); it != cObjectRegistry.end() )
    {
        auto& [_, ptr] = *it;
        return ptr;
    }
    return {};
}

} // namespace MR::detail

namespace MR
{

} // namespace MR
