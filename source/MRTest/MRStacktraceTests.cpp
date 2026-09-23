#include "MRMesh/MRMeshFwd.h"
#ifndef __EMSCRIPTEN__
#include "MRMesh/MRStacktrace.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, Stacktrace )
{
    const auto stacktrace = getCurrentStacktrace();
    spdlog::info( "Test stacktrace:\n{}", stacktrace );
    EXPECT_FALSE( stacktrace.empty() );
#if !defined NDEBUG && __cpp_lib_stacktrace >= 202011 && ( defined _WIN32 || defined MR_USE_STD_STACKTRACE )
    // std::stacktrace reports source files and lines of the frames with debug info
    EXPECT_NE( stacktrace.find( "MRStacktraceTests.cpp" ), std::string::npos );
#endif
}

} //namespace MR
#endif //!__EMSCRIPTEN__
