#pragma once

#ifndef MR_USE_STD_EXPECTED
// #include <version>
// #if __cpp_lib_expected >= 202211
// Currently not using `std::expected` for simplicity, because:
// 1. Clang 18 doesn't support libstdc++'s `std::expected`, which is a problem for the Python bindings. This got fixed in Clang 19.
// 2. `MRMeshDotNet` can't use `std::expected` too.
// In theory both can be fixed by defining `MR_DOT_NET_BUILD`.
#define MR_USE_STD_EXPECTED 0
#endif

#if MR_USE_STD_EXPECTED

#include <expected>

#else // !MR_USE_STD_EXPECTED

#ifndef MR_NODISCARD_TL_EXPECTED
// declare tl::expected as nodiscard
#define MR_NODISCARD_TL_EXPECTED 1
#endif

#if MR_NODISCARD_TL_EXPECTED
// fix unity builds
#include "MRSuppressWarning.h"
MR_SUPPRESS_WARNING_PUSH
MR_SUPPRESS_WARNING( "-Wattributes", 5240 )
namespace tl { template <class T, class E> class [[nodiscard]] expected; }
MR_SUPPRESS_WARNING_POP
#endif

#include <tl/expected.hpp>
/// we have C++/CLI project MRMeshDotNet which doesn't support std::expected
/// So we have to wrap tl::expected with this class in std namespace for correct linking
#ifdef MR_DOT_NET_BUILD
namespace std
{
template<typename T, typename E>
class expected : public tl::expected<T, E>
{
    using tl::expected<T, E>::expected;
};

template <class E>
inline auto unexpected( E &&e )
{
    return tl::make_unexpected( std::forward<E>( e ) );
}
}
#endif
#endif

