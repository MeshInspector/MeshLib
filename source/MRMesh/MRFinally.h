#pragma once

#include "MRMacros.h"

#include <utility>

//! Usage: `MR_FINALLY{...};`. Runs the code in braces when exiting the current scope, either normally or via an exception.
//! To run it only on exception, set a flag after the throwing code:
//! `bool ok = false; MR_FINALLY{ if ( !ok ) ...; }; ...; ok = true;`
#define MR_FINALLY DETAIL_MR_FINALLY( ScopeGuard )

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
