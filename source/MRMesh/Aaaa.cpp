#ifndef __EMSCRIPTEN__

#include <fmt/format.h>
#include "MRGTest.h"

const auto aaa = fmt::format( "{}", FMT_VERSION );

TEST( MRMesh, aaa )
{
    std::cout << aaa << std::endl;
}

#endif
