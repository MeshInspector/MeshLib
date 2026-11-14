#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

// MR_PREFETCH( p ) hints the processor to load the cache line of p for reading
#if !defined(_MSC_VER) || defined(__clang__)
#define MR_PREFETCH( p ) __builtin_prefetch( p )
#elif defined(_M_ARM64)
#define MR_PREFETCH( p ) __prefetch( p )
#else
#define MR_PREFETCH( p ) _mm_prefetch( (const char *)( p ), _MM_HINT_T0 )
#endif
