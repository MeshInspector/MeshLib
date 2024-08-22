#pragma once

#include "MRMeshFwd.h"

#ifndef __EMSCRIPTEN__

#include <string>

#ifdef _WIN32
// it is tricky to use std::stacktrace on other systems: https://stackoverflow.com/q/78395268/7325599
#include <version>
#if __cpp_lib_stacktrace >= 202011
#pragma message("std::stacktrace is available")
#include <stacktrace>
#else
#pragma message("std::stacktrace is NOT available, using boost::stacktrace instead")
#include <boost/stacktrace.hpp>
#endif
#else //not _WIN32
#include <boost/stacktrace.hpp>
#endif //_WIN32

namespace MR
{

/// returns string representation of the current stacktrace;
/// the function is inlined, to put the code in any shared library;
/// if std::stacktrace is first called from MRMesh.dll then it is not unloaded propely
[[nodiscard]] inline std::string getCurrentStacktraceInline()
{
#if defined _WIN32 && __cpp_lib_stacktrace >= 202011
    return to_string( std::stacktrace::current() );
#else
    return to_string( boost::stacktrace::stacktrace() );
#endif
}

/// Print stacktrace on application crash
MRMESH_API void printStacktraceOnCrash();

} //namespace MR

#endif //!__EMSCRIPTEN__
