#pragma once

#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

[[noreturn]] void throwJsError( const std::string& msg );

emscripten::val toFloat32Array( const float* data, size_t count );
emscripten::val toUint32Array( const uint32_t* data, size_t count );

template <typename F>
auto guarded( F&& f ) -> decltype( f() )
{
    try
    {
        return f();
    }
    catch ( const std::exception& e )
    {
        throwJsError( e.what() );
    }
    catch ( ... )
    {
        throwJsError( "unknown error" );
    }
}
