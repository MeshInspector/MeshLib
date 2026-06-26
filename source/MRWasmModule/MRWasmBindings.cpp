#include "MRWasmBindings.h"

#include <emscripten/val.h>
#include <emscripten.h>

#include <cstdlib>

using namespace emscripten;

namespace Wasm
{

[[noreturn]] void throwJsError( const std::string& msg )
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( { throw new Error( UTF8ToString( $0 ) ); }, msg.c_str() );
#pragma clang diagnostic pop
    std::abort();
}

// Must copy out: a typed_memory_view aliases WASM memory and is detached when the
// heap grows, so it must never be retained or returned to JS.
val toFloat32Array( const float* data, size_t count )
{
    auto out = val::global( "Float32Array" ).new_( count );
    if ( count != 0 )
        out.call<void>( "set", val( typed_memory_view( count, data ) ) );
    return out;
}

val toUint32Array( const uint32_t* data, size_t count )
{
    auto out = val::global( "Uint32Array" ).new_( count );
    if ( count != 0 )
        out.call<void>( "set", val( typed_memory_view( count, data ) ) );
    return out;
}

std::function<bool( float )> jsToCppCallback( val cb )
{
    if ( cb.isUndefined() || cb.isNull() )
        return {};
    return [cb]( float progress ) -> bool
    {
        auto r = cb( progress );
        return r.isUndefined() ? true : r.as<bool>();
    };
}

}

int main()
{
    return 0;
}
