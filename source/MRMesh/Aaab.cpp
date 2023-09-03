#ifndef __EMSCRIPTEN__

#include <spdlog/spdlog.h>
#include <gtest/gtest.h>

void aab()
{
    spdlog::info( "SPDLOG_VERSION={}", SPDLOG_VERSION );
    spdlog::info( "FMT_VERSION={}", FMT_VERSION );
}

struct AAB
{
    AAB() { aab(); }
} sAAB;

TEST( MRMesh, aabtest )
{
    aab();
}

#endif
