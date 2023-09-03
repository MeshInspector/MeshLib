#ifndef __EMSCRIPTEN__

#if __GNUC__ == 13
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#include <spdlog/spdlog.h>
#include <gtest/gtest.h>

TEST( a, b ) {
    spdlog::info( "info" );
}

#endif
