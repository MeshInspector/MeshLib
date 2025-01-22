#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"

#include <version>
#ifndef MR_USE_STD_EXPECTED
// #if __cpp_lib_expected >= 202211
// Currently not using `std::expected` for simplicity, because:
// 1. Clang 18 doesn't support libstdc++'s `std::expected`, which is a problem for the Python bindings. This got fixed in Clang 19.
// 2. `MRMeshDotNet` can't use `std::expected` too.
// In theory both can be fixed by defining `MR_DOT_NET_BUILD`.
#define MR_USE_STD_EXPECTED 0
#endif

#if MR_USE_STD_EXPECTED
#include <expected>
#else
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

#include <string>

namespace MR
{

#if MR_USE_STD_EXPECTED || defined(MR_DOT_NET_BUILD)

template<class T, class E = std::string>
using Expected = std::expected<T, E>;

template <class E>
MR_BIND_IGNORE inline auto unexpected( E &&e )
{
    return std::unexpected( std::forward<E>( e ) );
}

#else

template<class T, class E = std::string>
using Expected = tl::expected<T, E>;

template <class E>
MR_BIND_IGNORE inline auto unexpected( E &&e )
{
    return tl::make_unexpected( std::forward<E>( e ) );
}

#endif

/// common message about user termination of an operation
MR_BIND_IGNORE inline std::string stringOperationCanceled()
{
    return "Operation was canceled";
}

/// returns Expected error with `stringOperationCanceled()`
MR_BIND_IGNORE inline auto unexpectedOperationCanceled()
{
    return MR::unexpected( stringOperationCanceled() );
}

/// common message about unknown file extension
MR_BIND_IGNORE inline std::string stringUnsupportedFileExtension()
{
    return "Unsupported file extension";
}

/// returns Expected error with `stringUnsupportedFileExtension()`
MR_BIND_IGNORE inline auto unexpectedUnsupportedFileExtension()
{
    return MR::unexpected( stringUnsupportedFileExtension() );
}

} //namespace MR
