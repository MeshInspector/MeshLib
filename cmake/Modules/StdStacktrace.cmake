# mr_use_std_stacktrace(<target>)
#
# Make <target> and its consumers use std::stacktrace (see MRMesh/MRStacktrace.h) if the toolchain provides it.
# libstdc++ keeps its implementation in the separate static library stdc++exp, so the check links a shared library
# against it; it follows the project's C++ standard and fails for C++20 builds and for libc++, which lack std::stacktrace.
# Windows uses std::stacktrace without extra libraries, so guard the call with if(NOT WIN32 AND NOT EMSCRIPTEN).
function(mr_use_std_stacktrace target)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS -fPIC)
  set(CMAKE_REQUIRED_LINK_OPTIONS -shared)
  set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
  check_cxx_source_compiles("
    #include <version>
    #if __cpp_lib_stacktrace < 202011
    #error no std::stacktrace
    #endif
    #include <stacktrace>
    int f() { return int( to_string( std::stacktrace::current() ).size() ); }
  " MR_STD_STACKTRACE_WITH_STDCXXEXP)
  if(MR_STD_STACKTRACE_WITH_STDCXXEXP)
    target_compile_definitions(${target} PUBLIC MR_USE_STD_STACKTRACE)
    target_link_libraries(${target} PUBLIC stdc++exp)
  endif()
endfunction()
