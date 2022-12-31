#pragma once

#include <functional>

namespace MR
{

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

/// returns a callback that maps [0,1] linearly into [from,to] in the call to \param p
inline ProgressCallback subprogress( ProgressCallback p, float from, float to )
{
    ProgressCallback res;
    if ( p )
        res = [p, from, to]( float v ) { return p( ( 1 - v ) * from + v * to ); };
    return res;
}

} //namespace MR
