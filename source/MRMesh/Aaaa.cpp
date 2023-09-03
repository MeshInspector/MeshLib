#ifndef __EMSCRIPTEN__

#include <fmt/format.h>
#include "MRGTest.h"

TEST( MRMesh, aaa )
{
    const auto a = fmt::format( "{}", FMT_VERSION );
    //std::cout << fmt::format( "1" ) << std::endl;
    std::cout << a << std::endl;
}

#endif
