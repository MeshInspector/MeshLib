#pragma once

#include "MRMeshFwd.h"

#include <boost/type_index.hpp>

namespace MR::detail
{

MRMESH_API void registerObject( const std::string& typeName, std::shared_ptr<void*> objectPtr );

MRMESH_API void unregisterObject( const std::string& typeName, void* objectPtr );

MRMESH_API std::vector<std::shared_ptr<void*>> getObjects( const std::string& typeName );

MRMESH_API std::shared_ptr<void*> getObject( const std::string& typeName );

template <typename T>
auto getTypeName()
{
    return boost::typeindex::type_id<T>().pretty_name();
}

template <typename T>
void registerObject( std::shared_ptr<T> object )
{
    auto* objectPtr = (void*)object.get();
    registerObject( getTypeName<T>(), std::shared_ptr<void*>( std::move( object ), objectPtr ) );
}

template <typename T>
void unregisterObject( T* objectPtr )
{
    unregisterObject( getTypeName<T>(), (void*)objectPtr );
}

template <typename T>
std::vector<std::shared_ptr<T>> getObjects()
{
    std::vector<std::shared_ptr<T>> results;
    for ( auto&& objectPtr : getObjects( getTypeName<T>() ) )
        results.emplace_back( std::move( objectPtr ), reinterpret_cast<T*>( objectPtr.get() ) );
    return results;
}

template <typename T>
std::shared_ptr<T> getObject()
{
    auto objectPtr = getObject( getTypeName<T>() );
    return { std::move( objectPtr ), reinterpret_cast<T*>( objectPtr.get() ) };
}

} // namespace MR::detail

namespace MR
{

class ObjectRegistry
{
public:
    template <typename T>
    static void add( std::shared_ptr<T> object )
    {
        detail::registerObject<T>( std::move( object ) );
    }

    template <typename T>
    static std::shared_ptr<T> get()
    {
        return detail::getObject<T>();
    }
};

} // namespace MR
