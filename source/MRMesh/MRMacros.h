#pragma once

// Those are generic helper macros that don't have their own headers.

// Convert to a string.
#define MR_STR(...) MR_STR_(__VA_ARGS__)
#define MR_STR_(...) #__VA_ARGS__

// Returns the argument unchanged.
#define MR_IDENTITY(...) __VA_ARGS__

// A helper for writing preprocessor loops.
#define MR_END(...) DETAIL_MR_END(__VA_ARGS__)
#define DETAIL_MR_END(...) __VA_ARGS__##_END

// Concat strings
#define MR_CONCAT( a, b ) MR_CONCAT_( a, b )
#define MR_CONCAT_( a, b ) a ## b

// If the compiler supports `requires`, expands to `requires(...)`. Otherwise to nothing.
// This is primarily useful for code that must be usable in Cuda, since everywhere else we're free to use C++20 and newer.
// While Clang 14 technically supports `requires`, we're getting a few weird issues with it (make a nested aggregate class,
//   in the enclosing class make a `MR::Vector` of it, observe that `std::default_initializable` gets baked as `false` on it,
//   disabling some member functions such as `.resize()`).
#if __cpp_concepts && __has_include(<concepts>) && !(defined(__clang__) && __clang_major__ <= 14) && !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 12)
#   define MR_REQUIRES_IF_SUPPORTED(...) requires(__VA_ARGS__)
#   define MR_HAS_REQUIRES 1
#else
#   define MR_REQUIRES_IF_SUPPORTED(...)
#   define MR_HAS_REQUIRES 0
#endif


#ifdef _MSC_VER
#define MR_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]] // The unprefixed version has no effect on MSVC.
#else
#define MR_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif


// Are we using the old buggy MSVC preprocessor?
#if defined(_MSC_VER) && !defined(__clang__) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL == 1)
#define MR_LEGACY_MSVC_PREPROCESSOR 1
#else
#define MR_LEGACY_MSVC_PREPROCESSOR 0
#endif

// `MR_TRIM_LEADING_COMMA(,a,b,c)` returns `a,b,c`.
#if MR_LEGACY_MSVC_PREPROCESSOR
#define MR_TRIM_LEADING_COMMA(...) DETAIL_MR_TRIM_LEADING_COMMA_ DETAIL_MR_TRIM_LEADING_COMMA_DEFER(__VA_ARGS__)
#define DETAIL_MR_TRIM_LEADING_COMMA_DEFER
#else
#define MR_TRIM_LEADING_COMMA(...) DETAIL_MR_TRIM_LEADING_COMMA_(__VA_ARGS__)
#endif
#define DETAIL_MR_TRIM_LEADING_COMMA_(x, ...) DETAIL_MR_TRIM_LEADING_COMMA_EMPTY(x) __VA_ARGS__
#define DETAIL_MR_TRIM_LEADING_COMMA_EMPTY()