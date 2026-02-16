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
#define MR_CONCAT_( a, b ) a##b

#define MR_CONCAT3( a, b, c ) MR_CONCAT3_( a, b, c )
#define MR_CONCAT3_( a, b, c ) a##b##c

// If the compiler supports `requires`, expands to `requires(...)`. Otherwise to nothing.
// This is primarily useful for code that must be usable in Cuda, since everywhere else we're free to use C++20 and newer.
// While Clang 14 technically supports `requires`, we're getting a few weird issues with it (make a nested aggregate class,
//   in the enclosing class make a `MR::Vector` of it, observe that `std::default_initializable` gets baked as `false` on it,
//   disabling some member functions such as `.resize()`).
#if __cpp_concepts && __has_include(<concepts>) && !(defined(__clang__) && __clang_major__ <= 14) && !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 12)
#   define MR_HAS_REQUIRES 1
#   define MR_REQUIRES_IF_SUPPORTED(...) requires(__VA_ARGS__)
#   define MR_SAME_TYPE_TEMPLATE_PARAM(target_, name_) std::same_as<target_> name_ = target_
#else
#   define MR_HAS_REQUIRES 0
#   define MR_REQUIRES_IF_SUPPORTED(...)
#   define MR_SAME_TYPE_TEMPLATE_PARAM(target_, name_) typename name_ = target_
#endif


#ifdef _MSC_VER
#define MR_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]] // The unprefixed version has no effect on MSVC.
#else
#define MR_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif


// Those attributes can be used to help Clang emit more warnings. We also use them in generated bindings to control keep-alive (object lifetime extension). At the time of writing, this is only used in C#.
// `MR_LIFETIMEBOUND` can be placed either on a function parameter (immediately after the parameter name), or on `this` by placing it after the method parameter list (after `const` if any).
//   It indicates that the return value preserves a reference to this parameter (or to `this`).
//   For this attribute, constructors are considered to return the object that they construct.
// `MR[_THIS]_LIFETIME_CAPTURE_BY(x)` is usable in the same places.
//   (The `THIS` version can only be placed after a method parameter list, and the non-`THIS` version can only be placed on parameters. This limitation exists because we have to support
//     versions of Clang older than 20 in our parser. Once we update to Clang 20 or newer, we can merge them. TODO: Do this when updating Clang.)
//   `x` must either be the name of another parameter of the same function, or `this`.
//   It indicates that the reference to the parameter (that has the attribute) is saved not in the return value, but instead in the specified parameter (the argument of the attribute, which could be `this`).
//   In constructors, `MR_LIFETIME_CAPTURE_BY(this)` has the same effect as `MR_LIFETIMEBOUND`, since `this` and the returned object are considered to be the same thing.
// See Clang docs for more details:
//   https://clang.llvm.org/docs/AttributeReference.html#lifetimebound
//   https://clang.llvm.org/docs/AttributeReference.html#lifetime-capture-by
#ifdef __clang__
#  define MR_LIFETIMEBOUND [[clang::lifetimebound]]
#  if __clang_major__ >= 20 // Added in Clang 20.
#    define MR_LIFETIME_CAPTURE_BY(x) [[clang::lifetime_capture_by(x)]]
#    define MR_THIS_LIFETIME_CAPTURE_BY(x) [[clang::lifetime_capture_by(x)]]
#  else
#    define MR_LIFETIME_CAPTURE_BY(x) [[clang::annotate("mrbind::lifetime_capture_by=" #x)]]
#    define MR_THIS_LIFETIME_CAPTURE_BY(x) [[clang::annotate_type("mrbind::lifetime_capture_by=" #x)]]
#  endif
#else
#  define MR_LIFETIMEBOUND
#  define MR_LIFETIME_CAPTURE_BY(x)
#  define MR_THIS_LIFETIME_CAPTURE_BY(x)
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
