#include <spdlog/spdlog.h>
#include "MRGTest.h"

void aab()
{
    spdlog::info( "SPDLOG_VERSION={}", SPDLOG_VERSION );
    spdlog::info( "FMT_VERSION={}", FMT_VERSION );
}

TEST( MRMesh, aab )
{
    aab();
}
