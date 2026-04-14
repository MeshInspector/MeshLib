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

#include <parallel_hashmap/phmap_config.h>
#ifdef __aarch64__
// force on Clang for ABI compatibility with GCC:
// https://github.com/greg7mdp/parallel-hashmap/issues/289
#undef PHMAP_HAVE_INTRINSIC_INT128
#define PHMAP_HAVE_INTRINSIC_INT128 1
#endif
#include <parallel_hashmap/phmap.h>

MR_SUPPRESS_WARNING_POP
