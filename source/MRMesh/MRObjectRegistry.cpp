#include "MRObjectRegistry.h"

#include <mutex>

namespace
{

auto findValue( const auto& container, const auto& value )
{
    for ( auto it = container.begin(); it != container.end(); ++it )
        if ( it->second == value )
            return it;
    return container.end();
}

} // namespace

namespace MR::detail
{

GenericObjectRegistry& GenericObjectRegistry::get( const std::type_index& type )
{
    static std::map<std::type_index, GenericObjectRegistry> storage = {};
    static std::shared_mutex mutex = {};

    {
        std::shared_lock readLock( mutex );
        if ( auto it = storage.find( type ); it != storage.end() )
            return it->second;
    }

    std::unique_lock writeLock( mutex );
    return storage[type];
}

bool GenericObjectRegistry::add( std::string id, std::shared_ptr<void> object, int priority )
{
    std::unique_lock lock( mutex_ );

    if ( auto it = map_.find( id ); it != map_.end() )
        return false;

    priorityQueue_.emplace( priority, id );
    map_.emplace( std::move( id ), std::move( object ) );
    return true;
}

void GenericObjectRegistry::remove( const std::string& id )
{
    std::unique_lock lock( mutex_ );

    if ( auto rit = map_.find( id ); rit != map_.end() )
    {
        if ( auto pqit = findValue( priorityQueue_, id ); pqit != priorityQueue_.end() )
            priorityQueue_.erase( pqit );

        map_.erase( rit );
    }
}

void GenericObjectRegistry::remove( const std::shared_ptr<void>& object )
{
    std::unique_lock lock( mutex_ );

    if ( auto rit = findValue( map_, object ); rit != map_.end() )
    {
        const auto& id = rit->first;
        if ( auto pqit = findValue( priorityQueue_, id ); pqit != priorityQueue_.end() )
            priorityQueue_.erase( pqit );

        map_.erase( rit );
    }
}

std::shared_ptr<void> GenericObjectRegistry::findObject( const std::string& id ) const
{
    std::shared_lock lock( mutex_ );

    if ( auto rit = map_.find( id ); rit != map_.end() )
        return rit->second;
    return {};
}

std::shared_ptr<void> GenericObjectRegistry::getTopObject() const
{
    std::shared_lock lock( mutex_ );

    if ( priorityQueue_.empty() )
        return {};
    return map_.at( priorityQueue_.begin()->second );
}

std::vector<std::shared_ptr<void>> GenericObjectRegistry::getAllObjects() const
{
    std::shared_lock lock( mutex_ );

    std::vector<std::shared_ptr<void>> results;
    results.reserve( map_.size() );
    for ( const auto& [_, id] : priorityQueue_ )
        results.emplace_back( map_.at( id ) );
    return results;
}

} // namespace MR::detail

namespace MR
{

} // namespace MR
