#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include "MRPch/MRExpected.h"
#include <string>

namespace MR
{

#if MR_USE_STD_EXPECTED || defined(MR_DOT_NET_BUILD)

template<class T, class E = std::string>
using Expected = std::expected<T, E>;

template<class E = std::string>
using Unexpected = std::unexpected<E>;

template <class E>
MR_BIND_IGNORE inline auto unexpected( E &&e )
{
    return std::unexpected( std::forward<E>( e ) );
}

#else

template<class T, class E = std::string>
using Expected = tl::expected<T, E>;

template<class E = std::string>
using Unexpected = tl::unexpected<E>;

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

/// common message prefix about unsupported file format
MR_BIND_IGNORE inline std::string stringUnsupportedFileFormat()
{
    return "Unsupported file format";
}

/// returns Expected error with `stringUnsupportedFileFormat()`
MR_BIND_IGNORE inline auto unexpectedUnsupportedFileFormat()
{
    return MR::unexpected( stringUnsupportedFileFormat() );
}

/// Exits the current function with an error if the given expression contains an error.
#define MR_RETURN_IF_UNEXPECTED( expr ) \
    if ( auto&& res = ( expr ); !res ) \
        return unexpected( std::move( res.error() ) );

} //namespace MR
