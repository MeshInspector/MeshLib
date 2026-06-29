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
