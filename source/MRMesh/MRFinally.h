#pragma once

#include "MRMacros.h"

#ifdef __cpp_exceptions
#include <exception>
#endif
#include <utility>

//! Usage: `MR_FINALLY{...};`. Runs the code in braces when exiting the current scope, either normally or via an exception.
#define MR_FINALLY DETAIL_MR_FINALLY( ScopeGuard )

#ifdef __cpp_exceptions

//! Usage: `MR_FINALLY_ON_SUCCESS{...};`. Runs the code in braces when exiting the current scope without an exception.
#define MR_FINALLY_ON_SUCCESS DETAIL_MR_FINALLY( ExceptionScopeGuard<true>::Type )

//! Usage: `MR_FINALLY_ON_THROW{...};`. Runs the code in braces when exiting the current scope due to an exception.
#define MR_FINALLY_ON_THROW DETAIL_MR_FINALLY( ExceptionScopeGuard<false>::Type )

#else // If no exceptions.

//! When exceptions are disabled, this is equivalent to `MR_FINALLY`.
#define MR_FINALLY_ON_SUCCESS MR_FINALLY
//! When exceptions are disabled, this is a noop.
#define MR_FINALLY_ON_THROW (void)[&]()

#endif // If no exceptions.

#define DETAIL_MR_FINALLY( type_ ) \
    auto MR_CONCAT( _mrScopeGuard, __COUNTER__ ) = ::MR::detail::MakeScopeGuard<::MR::detail::type_>{} ->* [&]() -> void

namespace MR::detail
{

template <typename F>
class ScopeGuard
{
    F func;

public:
    ScopeGuard( F&& func ) : func( std::move( func ) ) {}
    ScopeGuard( const ScopeGuard& ) = delete;
    ScopeGuard& operator=( const ScopeGuard& ) = delete;
    ~ScopeGuard() { func(); }
};

#ifdef __cpp_exceptions
template <bool Success>
struct ExceptionScopeGuard
{
    template <typename F>
    class Type
    {
        F func;
        int ex = std::uncaught_exceptions();

    public:
        Type( F&& func ) : func( std::move( func ) ) {}
        Type( const Type& ) = delete;
        Type& operator=( const Type& ) = delete;
        ~Type() noexcept( !Success )
        {
            if ( ( ex == std::uncaught_exceptions() ) == Success )
                func();
        }
    };
};
#endif

template <template <typename> typename T>
struct MakeScopeGuard
{
    template <typename F>
    T<F> operator->*( F&& func )
    {
        return T<F>( std::move( func ) );
    }
};

} // namespace MR::detail
