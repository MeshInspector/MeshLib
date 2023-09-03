#ifndef __EMSCRIPTEN__

#include <spdlog/spdlog.h>
#include <gtest/gtest.h>
#include <iostream>

TEST( MRMesh, aaatest )
{
    auto x = fmt::format( "FMT_VERSION={}", FMT_VERSION );
    std::cout << x << std::endl;
}

#endif
