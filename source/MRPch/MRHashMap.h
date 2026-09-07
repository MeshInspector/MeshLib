#pragma once

#include "MRSuppressWarning.h"

MR_SUPPRESS_WARNING_PUSH
#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wshift-count-overflow"
#elif __clang_major__ >= 21
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#pragma warning(disable: 4574) // '__has_feature' defined to 0 by Eigen/spdlog shim; phmap's #ifdef is benign

#include <parallel_hashmap/phmap_config.h>
// force on Clang for ABI compatibility with GCC:
// https://github.com/greg7mdp/parallel-hashmap/issues/289
// Not on Windows arm64: there MSVC builds MeshLib and Clang only the bindings, and MSVC
// has no __int128, so forcing it on would create the very mismatch this avoids on Linux
// (Clang defines __aarch64__ even when targeting the MSVC ABI, MSVC only _M_ARM64).
#if defined( __aarch64__ ) && !defined( _WIN32 )
#undef PHMAP_HAVE_INTRINSIC_INT128
#define PHMAP_HAVE_INTRINSIC_INT128 1
#endif
#include <parallel_hashmap/phmap.h>

MR_SUPPRESS_WARNING_POP
