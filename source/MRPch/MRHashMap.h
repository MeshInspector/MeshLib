#pragma once

#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wshift-count-overflow"
#endif
#include <parallel_hashmap/phmap.h>
#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif
