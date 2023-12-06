#pragma once

#include "MRMeshFwd.h"

#include <version>
#if __cpp_lib_expected >= 202211
#include <expected>
#else
#include <tl/expected.hpp>
#endif

#include <string>

namespace MR
{

#if __cpp_lib_expected >= 202211

template<class T, class E = std::string>
using Expected = std::expected<T, E>;

template <class E>
inline auto unexpected( E &&e )
{
    return std::unexpected( std::forward<E>( e ) );
}

#else

template<class T, class E = std::string>
using Expected = tl::expected<T, E>;

template <class E>
inline auto unexpected( E &&e )
{
    return tl::make_unexpected( std::forward<E>( e ) );
}

#endif

/// return type for a void function that can produce an error string
using VoidOrErrStr = Expected<void, std::string>;

/// Common operation canceled line for all
inline std::string stringOperationCanceled()
{
    return "Operation was canceled";
}

/// Returns Expected error with `stringOperationCanceled()`
inline auto unexpectedOperationCanceled()
{
    return unexpected( stringOperationCanceled() );
}

} //namespace MR
