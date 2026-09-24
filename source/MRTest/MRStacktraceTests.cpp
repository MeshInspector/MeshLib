#include "MRMesh/MRMeshFwd.h"
#ifndef __EMSCRIPTEN__
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
#if !defined NDEBUG && defined _WIN32
    // both std::stacktrace and Boost.Stacktrace report source files and lines of the frames with debug info on Windows
    EXPECT_NE( stacktrace.find( "MRStacktraceTests.cpp" ), std::string::npos );
#endif
}

} //namespace MR
#endif //!__EMSCRIPTEN__
