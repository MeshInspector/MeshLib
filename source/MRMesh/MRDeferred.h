#pragma once

#include <utility>

// TODO: move to separate header
#define MR_CONCAT_IMPL( A, B ) A##B
#define MR_CONCAT( A, B ) MR_CONCAT_IMPL( A, B )

namespace MR
{

/// helper class to call a function at its scope's exit (RAII-style)
template <typename F>
class DeferredFunction
{
    F f_;

public:
    DeferredFunction( F&& f )
        : f_( std::forward<F>( f ) )
    {
        //
    }
    ~DeferredFunction()
    {
        f_();
    }

    DeferredFunction( const DeferredFunction& ) = delete;
    DeferredFunction( DeferredFunction&& ) = delete;
    DeferredFunction& operator =( const DeferredFunction& ) = delete;
    DeferredFunction& operator =( DeferredFunction&& ) = delete;
};

template <typename F>
DeferredFunction<F> defer( F&& f )
{
    return { std::forward<F>( f ) };
}

} // namespace MR

#define MR_DEFER( F ) [[maybe_unused]] const auto MR_CONCAT( deferred, __LINE__ ) = MR::defer( F );

#define MR_DEFER_INLINE( F ) MR_DEFER( [&] { F ; } )
