#ifndef __EMSCRIPTEN__

#include <spdlog/spdlog.h>
#include <gtest/gtest.h>

TEST( a, c ) {
    spdlog::info( "info {}", 1 );
}

#endif
