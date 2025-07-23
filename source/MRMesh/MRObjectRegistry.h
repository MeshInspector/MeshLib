#pragma once

#include "MRMeshFwd.h"

#include <map>
#include <shared_mutex>
#include <typeindex>

namespace MR::detail
{

class GenericObjectRegistry
{
public:
    MRMESH_API static GenericObjectRegistry& get( const std::type_index& type );

    MRMESH_API bool add( std::string id, std::shared_ptr<void> object, int priority = 0 );

    MRMESH_API void remove( const std::string& id );
    MRMESH_API void remove( const std::shared_ptr<void>& object );

    MRMESH_API std::shared_ptr<void> findObject( const std::string& id ) const;

    MRMESH_API std::shared_ptr<void> getTopObject() const;

    MRMESH_API std::vector<std::shared_ptr<void>> getAllObjects() const;

private:
    std::unordered_map<std::string, std::shared_ptr<void>> map_;
    std::multimap<int, std::string> priorityQueue_;
    mutable std::shared_mutex mutex_;
};

} // namespace MR::detail

namespace MR
{

template <typename T>
class ObjectRegistry
{
public:
    static bool add( std::string id, std::shared_ptr<T> object, int priority = 0 )
    {
        static auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        return registry.add( std::move( id ), cast_<void>( std::move( object ) ), priority );
    }

    static void remove( const std::string& id )
    {
        static auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        registry.remove( id );
    }
    static void remove( const std::shared_ptr<T>& objectPtr )
    {
        static auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        registry.remove( cast_<void>( objectPtr ) );
    }

    static std::shared_ptr<T> findObject( const std::string& id )
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        return cast_<T>( registry.findObject( id ) );
    }

    static std::shared_ptr<T> getTopObject()
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        return cast_<T>( registry.getTopObject() );
    }

    static std::vector<std::shared_ptr<T>> getAllObjects()
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        std::vector<std::shared_ptr<T>> results;
        for ( auto&& object : registry.getAllObjects() )
            results.emplace_back( cast_<T>( std::move( object ) ) );
        return results;
    }

private:
    template <typename To, typename From>
    static std::shared_ptr<To> cast_( std::shared_ptr<From>&& object )
    {
        auto* ptr = object.get();
        return { std::forward<std::shared_ptr<From>>( object ), reinterpret_cast<To*>( ptr ) };
    }
};

} // namespace MR
