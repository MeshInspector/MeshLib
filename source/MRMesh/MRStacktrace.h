#pragma once

#include "MRMeshFwd.h"

#ifndef __EMSCRIPTEN__

#include <string>

#include <version>
// with libstdc++, std::stacktrace needs an extra library (https://stackoverflow.com/q/78395268/7325599),
// which cmake/Modules/StdStacktrace.cmake links
#if __cpp_lib_stacktrace >= 202011
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
#if __cpp_lib_stacktrace >= 202011
    return to_string( std::stacktrace::current() );
#else
    return to_string( boost::stacktrace::stacktrace() );
#endif
}

/// Print stacktrace on application crash
MRMESH_API void printStacktraceOnCrash();

} //namespace MR

#endif //!__EMSCRIPTEN__
