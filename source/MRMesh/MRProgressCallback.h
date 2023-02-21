#pragma once

#include <functional>

namespace MR
{

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

/// safely invokes \param cv with given value; just returning true for empty callback
inline bool reportProgress( ProgressCallback cb, float v )
{
    if ( cb )
        return cb( v );
    return true;
}

/// safely invokes \param cv with given value if \param counter is divisible by \param divider (preferably a power of 2);
/// just returning true for empty callback
inline bool reportProgress( ProgressCallback cb, float v, size_t counter, int divider )
{
    if ( cb && ( counter % divider == 0 ) )
        return cb( v );
    return true;
}

/// safely invokes \param cv with the value produced by given functor;
/// just returning true for empty callback and not evaluating the function
template<typename F>
inline bool reportProgress( ProgressCallback cb, F && f )
{
    if ( cb )
        return cb( f() );
    return true;
}

/// safely invokes \param cv with the value produced by given functor if \param counter is divisible by \param divider (preferably a power of 2);
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
    ProgressCallback res;
    if ( cb )
        res = [cb, from, to]( float v ) { return cb( ( 1 - v ) * from + v * to ); };
    return res;
}

/// returns a callback that maps the value with given function \param f before calling \param cb (which can be empty)
template<typename F>
inline ProgressCallback subprogress( ProgressCallback cb, F && f )
{
    ProgressCallback res;
    if ( cb )
        res = [cb, f = std::forward<F>( f )]( float v ) { return cb( f( v ) ); };
    return res;
}

} //namespace MR
