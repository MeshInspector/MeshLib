#pragma once

#include "MRMeshFwd.h"

#include <version>
#ifdef __cpp_lib_expected
#include <expected>
#else
#include <tl/expected.hpp>
#endif

#include <string>

namespace MR
{

#ifdef __cpp_lib_expected

template<class T, class E>
using Expected = std::expected<T,E>;

template <class E>
inline auto unexpected( E &&e )
    { return std::unexpected( std::forward<E>( e ) ); }

#else

template<class T, class E>
using Expected = tl::expected<T,E>;

template <class E>
inline auto unexpected( E &&e )
    { return tl::make_unexpected( std::forward<E>( e ) ); }

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
