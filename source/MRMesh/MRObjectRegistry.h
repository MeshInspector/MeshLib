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

    MRMESH_API std::shared_ptr<void> find( const std::string& id ) const;

    MRMESH_API std::vector<std::pair<std::string, std::shared_ptr<void>>> get() const;

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

    static std::shared_ptr<T> find( const std::string& id )
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        return cast_<T>( registry.find( id ) );
    }

    static std::vector<std::pair<std::string, std::shared_ptr<void>>> get()
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        std::vector<std::shared_ptr<T>> results;
        for ( auto&& [id, object] : registry.get() )
            results.emplace_back( id, cast_<T>( std::move( object ) ) );
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

template <typename T>
class ObjectFactoryRegistry
{
public:
    using Factory = std::function<std::shared_ptr<T> ( void )>;

    static bool add( std::string id, Factory&& factory, int priority = 0 )
    {
        static auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        auto object = std::make_shared<Factory>( std::forward<Factory>( factory ) );
        return registry.add( std::move( id ), cast_<void>( std::move( object ) ), priority );
    }

    static void remove( const std::string& id )
    {
        static auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        registry.remove( id );
    }

    static std::shared_ptr<Factory> find( const std::string& id )
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        return cast_<Factory>( registry.find( id ) );
    }

    static std::shared_ptr<T> make( const std::string& id )
    {
        static const auto& registry = detail::GenericObjectRegistry::get( typeid( T ) );
        auto factory = cast_<Factory>( registry.find( id ) );
        return factory ? (*factory)() : std::shared_ptr<T>{};
    }

private:
    template <typename To, typename From>
    static std::shared_ptr<To> cast_( std::shared_ptr<From>&& object )
    {
        auto* ptr = object.get();
        return { std::forward<std::shared_ptr<From>>( object ), reinterpret_cast<To*>( ptr ) };
    }
};

#define MR_FACTORY( Interface, Class ) [] () -> std::shared_ptr<Interface> { return std::make_shared<Class>(); }

#define MR_REGISTER_FACTORY( Interface, Class, Name, ... ) MR::ObjectFactoryRegistry<Interface>::add( Name, MR_FACTORY( Interface, Class ) __VA_OPT__(,) __VA_ARGS__ )

} // namespace MR
