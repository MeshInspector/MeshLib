#pragma once

#include "MRMeshFwd.h"

#ifndef __EMSCRIPTEN__

#include <string>

#include <version>
// on systems other than Windows, std::stacktrace needs extra flags and libraries: https://stackoverflow.com/q/78395268/7325599
// MRMesh/CMakeLists.txt links them and defines MR_USE_STD_STACKTRACE where they are available
#if __cpp_lib_stacktrace >= 202011 && ( defined _WIN32 || defined MR_USE_STD_STACKTRACE )
#ifdef _MSC_VER
#pragma message("std::stacktrace is available")
#endif
#include <stacktrace>
#else
#ifdef _MSC_VER
#pragma message("std::stacktrace is NOT available, using boost::stacktrace instead")
#endif
#include <boost/stacktrace.hpp>
#endif

namespace MR
{

/// returns string representation of the current stacktrace;
/// the function is inlined, to put the code in any shared library;
/// if std::stacktrace is first called from MRMesh.dll then it is not unloaded propely
[[nodiscard]] inline std::string getCurrentStacktraceInline()
{
#if __cpp_lib_stacktrace >= 202011 && ( defined _WIN32 || defined MR_USE_STD_STACKTRACE )
    return to_string( std::stacktrace::current() );
#else
    return to_string( boost::stacktrace::stacktrace() );
#endif
}

/// Print stacktrace on application crash
MRMESH_API void printStacktraceOnCrash();

} //namespace MR

#endif //!__EMSCRIPTEN__
