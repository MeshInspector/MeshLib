# this file must be included AFTER the `project' command because it relies on the detected compiler information

set(MR_PCH_DEFAULT OFF)
# for macOS, GCC, and Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
  set(MR_PCH_DEFAULT ON)
ELSEIF(MSVC)
  set(MR_PCH_DEFAULT ON)
#ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#  set(MR_PCH_DEFAULT ON)
ENDIF()
set(MR_PCH ${MR_PCH_DEFAULT} CACHE BOOL "Enable precompiled headers")
IF(MR_PCH AND NOT MR_EMSCRIPTEN AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()
message("MR_PCH=${MR_PCH}")

# make link to fail if there are unresolved symbols (GCC and Clang)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,defs")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
ENDIF()

# Warnings and misc compiler settings.
IF(MSVC)
  # C++-specific flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   /DImDrawIdx=unsigned /D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING /D_SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING /D_SILENCE_CXX23_ALIGNED_STORAGE_DEPRECATION_WARNING /D_SILENCE_CXX23_DENORM_DEPRECATION_WARNING /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")

  # Common C/C++ flags:

  set(MESHLIB_COMMON_C_CXX_FLAGS "/utf-8 /fp:precise /permissive- /Zc:wchar_t /Zc:forScope /Zc:inline /DNOMINMAX /D_CRT_SECURE_NO_DEPRECATE")

  # Vcpkg automatically adds `/external:W0`, but we duplicate it here because it somehow doesn't propagate to Lazperf.
  set(MESHLIB_COMMON_C_CXX_FLAGS "${MESHLIB_COMMON_C_CXX_FLAGS} /W4 /WX /external:W0 /external:env:INCLUDE")

  # Following warnings are silenced:
  # !! NOTE: Sync this list with `common.props` !!
  #   warning C4061: enumerator V in switch of enum E is not explicitly handled by a case label
  #   warning C4250: 'class1': inherits 'class2' via dominance
  #   warning C4324: structure was padded due to alignment specifier
  #   warning C4365: conversion from 'unsigned int' to 'int', signed/unsigned mismatch
  #   warning C4371: layout of class may have changed from a previous version of the compiler due to better packing of member
  #   warning C4388: '<': signed/unsigned mismatch
  #   warning C4435: Object layout under /vd2 will change due to virtual base
  #   warning C4514: unreferenced inline function has been removed
  #   warning C4582: constructor is not implicitly called
  #   warning C4583: destructor is not implicitly called
  #   warning C4599: command line argument number N does not match precompiled header
  #   warning C4605: 'MACRO' specified on current command line, but was not specified when precompiled header was built
  #   warning C4623: default constructor was implicitly defined as deleted
  #   warning C4625: copy constructor was implicitly defined as deleted
  #   warning C4626: assignment operator was implicitly defined as deleted
  #   warning C4866: compiler may not enforce left-to-right evaluation order for call to
  #   warning C4668: MACRO is not defined as a preprocessor macro, replacing with '0' for '#if/#elif'
  #   warning C4686: possible change in behavior, change in UDT return calling convention
  #   warning C4710: function not inlined
  #   warning C4711: function selected for automatic inline expansion
  #   warning C4820: N bytes padding added after data member
  #   warning C4868: compiler may not enforce left-to-right evaluation order in braced initializer list
  #   warning C5026: move constructor was implicitly defined as deleted
  #   warning C5027: move assignment operator was implicitly defined as deleted
  #   warning C5031: #pragma warning(pop): likely mismatch, popping warning state pushed in different file
  #   warning C5039: pointer or reference to potentially throwing function passed to 'extern "C"' function under -EHc. Undefined behavior may occur if this function throws an exception.
  #   warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
  #   warning C5104: found 'L#x' in macro replacement list, did you mean 'L""#x'?
  #   warning C5105: macro expansion producing 'defined' has undefined behavior
  #   warning C5219: implicit conversion from 'int' to 'float', possible loss of data
  #   warning C5243: using incomplete class can cause potential one definition rule violation due to ABI limitation
  #   warning C5246: the initialization of a subobject should be wrapped in braces
  #   warning C5262: implicit fall-through occurs here; are you missing a break statement? Use [[fallthrough]] when a break statement is intentionally omitted between cases
  #   warning C5264: 'const' variable is not used
  #   warning C26451: Arithmetic overflow: Using operator '+' on a 4 byte value and then casting the result to a 8 byte value. Cast the value to the wider type before calling operator '+' to avoid overflow (io.2).
  # !! NOTE: Sync this list with `common.props` !!
  set(MESHLIB_COMMON_C_CXX_FLAGS "${MESHLIB_COMMON_C_CXX_FLAGS} /wd4061 /wd4250 /wd4324 /wd4365 /wd4371 /wd4388 /wd4435 /wd4514 /wd4582 /wd4583 /wd4599 /wd4605 /wd4623 /wd4625 /wd4626 /wd4668 /wd4686 /wd4710 /wd4711 /wd4820 /wd4866 /wd4868 /wd5026 /wd5027 /wd5031 /wd5039 /wd5045 /wd5104 /wd5105 /wd5219 /wd5243 /wd5246 /wd5262 /wd5264 /wd26451")
ELSE()
  set(MESHLIB_COMMON_C_CXX_FLAGS "${MESHLIB_COMMON_C_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors")

  IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(MESHLIB_COMMON_C_CXX_FLAGS "${MESHLIB_COMMON_C_CXX_FLAGS} -Wno-newline-eof")
  ENDIF()
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_COMMON_C_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MESHLIB_COMMON_C_CXX_FLAGS}")

IF(WIN32)
  IF(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
  ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  ENDIF()

  IF(${CMAKE_GENERATOR} MATCHES "^Visual Studio")
    # enable parallel build
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  ENDIF()
ENDIF()

IF(NOT EMSCRIPTEN AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF()

IF(MSVC)
  add_definitions(-DUNICODE -D_UNICODE)
  add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
ENDIF()

IF(NOT MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes")
ENDIF()

# This allows us to share bindings for C++ types across compilers (across GCC and Clang). Otherwise Pybind refuses
#   to share them because the compiler name and the ABI version number are different, even when there's no actual ABI incompatibility in practice.
# We allow customizing those so that our clients can prevent their modules from talking to ours, e.g. to provide their own simplified bindings
#   for our classes, to avoid having our modules as dependencies.
# Pass empty strings to those to avoid customizing them at all.
set(MESHLIB_PYBIND11_COMPILER_TYPE_STRING "_meshlib" CACHE STRING "")
set(MESHLIB_PYBIND11_BUILD_ABI_STRING "_meshlib" CACHE STRING "")
IF(NOT "${MESHLIB_PYBIND11_COMPILER_TYPE_STRING}" STREQUAL "")
  add_compile_definitions(PYBIND11_COMPILER_TYPE=\"${MESHLIB_PYBIND11_COMPILER_TYPE_STRING}\")
ENDIF()
IF(NOT "${MESHLIB_PYBIND11_BUILD_ABI_STRING}" STREQUAL "")
  add_compile_definitions(PYBIND11_BUILD_ABI=\"${MESHLIB_PYBIND11_BUILD_ABI_STRING}\")
ENDIF()

# Things for our patched pybind: --- [

# It's a good idea to have this match `PYTHON_MIN_VERSION` in `scripts/mrbind/generate.mk`.
# Here `0x030800f0` corresponds to 3.8 (ignore the `f0` suffix at the end, it just means a release version as opposed to alpha/beta/etc).
add_compile_definitions(Py_LIMITED_API=0x030800f0)

# It's a good idea to have this match the value specified in `scripts/mrbind/generate.mk`. See that file for the explanation.
add_compile_definitions(PYBIND11_INTERNALS_VERSION=5)

# This affects the naming of our pybind shims.
set(MESHLIB_PYBIND11_LIB_SUFFIX "meshlib" CACHE STRING "")
add_compile_definitions(PYBIND11_NONLIMITEDAPI_LIB_SUFFIX_FOR_MODULE=\"${MESHLIB_PYBIND11_LIB_SUFFIX}\")

# ] --- end things for our patched pybind

# Warn about ABI incompatibilities.
# GCC 12 fixed a bug, and this fix affects the ABI: https://github.com/gcc-mirror/gcc/commit/a37e8ce3b66325f0c6de55c80d50ac1664c3d0eb
# Because of this fix GCC 11 and older are incompatible with GCC 12+, and also with Clang that we use the build the Python bindings.
# This breaks the bindings on Ubuntu 20.04 (where we use GCC 10).
# This ABI change affects inheriting from certain classes with trailing padding, and the fix is always to add a dummy member variable at the end (mark it with
#   MR_BIND_IGNORE to hide from the bindings) to make sure there's no trailing padding. This affects only those bases that are aggregates and have default
#   member initializers.
# We can remove this flag when we stop supporting Ubuntu 20.04. In theory, Ubuntu 22.04 also uses GCC 11 by default, but it also has GCC 12, and on it we
#   use GCC 12 for Meshlib, so the resulting MeshLib library is probably incompatible with GCC 11 anyway, so we don't care about this bug there.
# We only enable this on GCC 12, because the next versions introduce more ABI changes that warn here, and we don't care about them (about much newer GCCs being
#   incompatible with GCC 11.)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wabi=16")
ENDIF()
# completely ignore "maybe-uninitialized" for GCC because of false positives
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109561
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=116090
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
ENDIF()

# more info: https://bugs.openjdk.org/browse/JDK-8244653
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11 AND CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
ENDIF()
