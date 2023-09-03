#ifndef __EMSCRIPTEN__

#include <spdlog/spdlog.h>
#include "MRGTest.h"

void aab()
{
    spdlog::info( "SPDLOG_VERSION={}", SPDLOG_VERSION );
    spdlog::info( "FMT_VERSION={}", FMT_VERSION );
}

struct AAB
{
    AAB() { aab(); }
} sAAB;

//TEST( MRMesh, aab )
//{
//    aab();
//}

#endif
