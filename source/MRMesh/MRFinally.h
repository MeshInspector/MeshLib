#pragma once

#include <exception>
#include <utility>

//! Usage: `MR_FINALLY{...};`. Runs the code in braces when exiting the current scope, either normally or via an exception.
#define MR_FINALLY \
    auto DETAIL_MR_FINALLY_NAME = ::MR::detail::MakeScopeGuard<::MR::detail::ScopeGuard>{} ->* [&]
//! Usage: `MR_FINALLY_ON_SUCCESS{...};`. Runs the code in braces when exiting the current scope without an exception.
#define MR_FINALLY_ON_SUCCESS \
    auto DETAIL_MR_FINALLY_NAME = ::MR::detail::MakeScopeGuard<::MR::detail::ExceptionScopeGuard<true>::Type>{} ->* [&]
//! Usage: `MR_FINALLY_ON_THROW{...};`. Runs the code in braces when exiting the current scope due to an exception.
#define MR_FINALLY_ON_THROW \
    auto DETAIL_MR_FINALLY_NAME = ::MR::detail::MakeScopeGuard<::MR::detail::ExceptionScopeGuard<false>::Type>{} ->* [&]

#define DETAIL_MR_FINALLY_NAME DETAIL_MR_FINALLY_CAT( _mrScopeGuard, __COUNTER__ )
#define DETAIL_MR_FINALLY_CAT( x, y ) DETAIL_MR_FINALLY_CAT_( x, y )
#define DETAIL_MR_FINALLY_CAT_( x, y ) x##y

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
        ~Type()
        {
            if ( ( ex == std::uncaught_exceptions() ) == Success )
                func();
        }
    };
};

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
