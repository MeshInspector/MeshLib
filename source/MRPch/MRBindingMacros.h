#pragma once

// Those macros help control Python bindings generated using MRBind.

// MR_PARSING_FOR_ANY_BINDINGS - gets defined when parsing the source code
// MR_COMPILING_ANY_BINDINGS - gets defined when compiling the resulting bindings

#ifdef MR_PARSING_FOR_ANY_BINDINGS

// Use to specify valid template arguments for templates (usually function templates).
// For example:
//   template <typename T> void foo(T) {...};
//   MR_BIND_TEMPLATE( void foo(int) )
//   MR_BIND_TEMPLATE( void foo(float) )
//
// As with `extern template`, you might need to use `foo<...>` instead of `foo` if the template parameters can't be deduced from the
// parameter types and the return type.
#define MR_BIND_TEMPLATE(...) extern template __VA_ARGS__;

// Mark a declaration with this to avoid generating a binding for it.
#define MR_BIND_IGNORE __attribute__((__annotate__("mrbind::ignore")))

// This is a specialized replacement for `MR_CANONICAL_TYPEDEFS()`, to be used on full template specializations, where that macro doesn't work.
#define MR_BIND_PREFERRED_NAME(...) __attribute__((__annotate__(DETAIL_MR_BIND_PREFERRED_NAME(mrbind::preferred_name=__VA_ARGS__))))
#define DETAIL_MR_BIND_PREFERRED_NAME(...) #__VA_ARGS__

#else
#define MR_BIND_TEMPLATE(...)
#define MR_BIND_IGNORE
#define MR_BIND_PREFERRED_NAME(...)
#endif

#ifdef MR_COMPILING_PB11_BINDINGS
// Put this inside of a class.
// Then when this class is used as a function parameter, calling that function will temporarily unlock GIL when it's called.
// This e.g. prevents deadlocks if your function calls Python lambdas.
// This is enabled by default for e.g. `std::function`, but not for classes that have them as members.
// This is a weak hint, some things can override it.
// Notice this being enabled only when `MR_COMPILING_ANY_BINDINGS` is defined, so this won't get parsed and appear in Python bindings themselves.
#define MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM using _prefer_gil_unlock_when_used_as_param = void;
#else
#define MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
#endif
