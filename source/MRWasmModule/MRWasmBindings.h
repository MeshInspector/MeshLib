#pragma once

#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

namespace Wasm
{

[[noreturn]] void throwJsError( const std::string& msg );

template <typename E>
auto unwrap( E&& e )
{
    if ( !e.has_value() )
        throwJsError( e.error() );
    if constexpr ( !std::is_void_v<typename std::remove_reference_t<E>::value_type> )
        return std::move( *e );
}

template <typename S> struct TypedArrayName;
template <> struct TypedArrayName<float>    { static constexpr const char* value = "Float32Array"; };
template <> struct TypedArrayName<uint32_t> { static constexpr const char* value = "Uint32Array";  };

template <typename S>
emscripten::val makeTypedArray( const S* data, size_t count )
{
    // The view aliases WASM heap memory and detaches on growth; copy out, never retain or return it.
    auto out = emscripten::val::global( TypedArrayName<S>::value ).new_( count );
    if ( count != 0 )
        out.call<void>( "set", emscripten::val( emscripten::typed_memory_view( count, data ) ) );
    return out;
}

template <typename V, typename S, size_t Arity>
emscripten::val packedToTypedArray( const V& v )
{
    using Elem = typename V::value_type;
    static_assert( std::is_trivially_copyable_v<Elem> && sizeof( Elem ) == Arity * sizeof( S ) );
    return makeTypedArray<S>( reinterpret_cast<const S*>( v.data() ), v.size() * Arity );
}

template <typename V, typename S, size_t Arity>
V packedFromTypedArray( emscripten::val arr )
{
    using Elem = typename V::value_type;
    static_assert( std::is_trivially_copyable_v<Elem> && sizeof( Elem ) == Arity * sizeof( S ) );
    const size_t len = arr["length"].as<size_t>();
    V v;
    v.vec_.resize( len / Arity );
    if ( len != 0 )
    {
        auto view = emscripten::val( emscripten::typed_memory_view(
            len, reinterpret_cast<S*>( v.data() ) ) );
        view.call<void>( "set", arr );
    }
    return v;
}

std::function<bool( float )> jsToCppCallback( emscripten::val cb );

}
