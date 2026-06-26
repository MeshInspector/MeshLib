#pragma once

#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

[[noreturn]] void throwJsError( const std::string& msg );

template <typename E>
auto unwrap( E&& e )
{
    if ( !e.has_value() )
        throwJsError( e.error() );
    if constexpr ( !std::is_void_v<typename std::remove_reference_t<E>::value_type> )
        return std::move( *e );
}

emscripten::val toFloat32Array( const float* data, size_t count );
emscripten::val toUint32Array( const uint32_t* data, size_t count );

std::function<bool( float )> jsToCppCallback( emscripten::val cb );
