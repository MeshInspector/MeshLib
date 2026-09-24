# mr_link_std_stacktrace(<target>)
#
# Link <target> and its consumers with stdc++exp, where libstdc++ keeps the implementation of std::stacktrace,
# which MRMesh/MRStacktrace.h uses whenever __cpp_lib_stacktrace is defined. The check links a shared library and
# follows the project's C++ standard; it fails for C++20 builds and for libc++, which lack std::stacktrace.
# Windows uses std::stacktrace without extra libraries, so guard the call with if(NOT WIN32 AND NOT EMSCRIPTEN).
function(mr_link_std_stacktrace target)
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
    target_link_libraries(${target} PUBLIC stdc++exp)
  endif()
endfunction()
