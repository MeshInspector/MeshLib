#pragma once

#include "MRMeshFwd.h"

#include <array>
#include <cassert>
#include <cmath>
#include <tuple>
#include <utility>

namespace MR
{

/// safely invokes \param cb with given value; just returning true for empty callback
inline bool reportProgress( ProgressCallback cb, float v )
{
    if ( cb )
        return cb( v );
    return true;
}

/// safely invokes \param cb with given value if \param counter is divisible by \param divider (preferably a power of 2);
/// just returning true for empty callback
inline bool reportProgress( ProgressCallback cb, float v, size_t counter, int divider )
{
    if ( cb && ( counter % divider == 0 ) )
        return cb( v );
    return true;
}

/// safely invokes \param cb with the value produced by given functor;
/// just returning true for empty callback and not evaluating the function
template<typename F>
inline bool reportProgress( ProgressCallback cb, F && f )
{
    if ( cb )
        return cb( f() );
    return true;
}

/// safely invokes \param cb with the value produced by given functor if \param counter is divisible by \param divider (preferably a power of 2);
/// just returning true for empty callback and not evaluating the function
template<typename F>
inline bool reportProgress( ProgressCallback cb, F && f, size_t counter, int divider )
{
    if ( cb && ( counter % divider == 0 ) )
        return cb( f() );
    return true;
}

/// returns a callback that maps [0,1] linearly into [from,to] in the call to \param cb (which can be empty)
inline ProgressCallback subprogress( ProgressCallback cb, float from, float to )
{
    assert( from <= to );
    ProgressCallback res;
    if ( cb )
        res = [cb = std::move( cb ), from, to]( float v ) { return cb( std::lerp( from, to, v ) ); };
    return res;
}

/// returns a callback that maps the value with given function \param f before calling \param cb (which can be empty)
template<typename F>
inline ProgressCallback subprogress( ProgressCallback cb, F && f )
{
    ProgressCallback res;
    if ( cb )
        res = [cb = std::move( cb ), f = std::forward<F>( f )]( float v ) { return cb( f( v ) ); };
    return res;
}

/// returns a callback that maps [0,1] linearly into [(index+0)/count,(index+1)/count] in the call to \param cb (which can be empty)
inline ProgressCallback subprogress( ProgressCallback cb, size_t index, size_t count )
{
    assert( index < count );
    if ( cb )
        return [cb = std::move( cb ), index, count] ( float v ) { return cb( ( (float)index + v ) / (float)count ); };
    else
        return {};
}

/// splits the given progress on (n+1) sub-progresses: [0,t1], [t1,t2], ... [tn,1],
/// where n given thresholds must be sorted: 0 <= t1 <= ... <= tn <= 1;
/// returns the sub-progresses in std::tuple with (n+1) elements
template <typename ...Ts>
auto splitProgress( const ProgressCallback& cb, Ts ... thresholds )
{
    constexpr size_t n = sizeof...( Ts );
    static_assert( n > 0, "at least one threshold is required" );
    const std::array<float, n + 2> bounds{ 0.0f, float( thresholds )..., 1.0f };
    return [&]<size_t ...I>( std::index_sequence<I...> )
    {
        return std::tuple{ subprogress( cb, bounds[I], bounds[I + 1] )... };
    }( std::make_index_sequence<n + 1>{} );
}

} //namespace MR
