#include "MRWasmBindings.h"

#include <emscripten/bind.h>
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

EMSCRIPTEN_BINDINGS( meshlib_common_val_types )
{
    register_type<Wasm::Float32ArrayVal>( Wasm::TypedArrayName<float>::value );
    register_type<Wasm::Uint32ArrayVal>( Wasm::TypedArrayName<uint32_t>::value );
    register_type<Wasm::Uint8ArrayVal>( Wasm::TypedArrayName<uint8_t>::value );
    register_type<Wasm::IndicesInputVal>( "readonly number[] | Uint32Array" );
}

int main()
{
    return 0;
}
