#pragma once

#include "MRMeshFwd.h"
#include "MRSignal.h"
#include <string>

namespace MR
{

/// activate this signal if you want to add some string in telemetry
extern MRMESH_API Signal<void( const std::string& )> TelemetrySignal;

/// integer binary logarithm:
///   0   -> 0
///   1   -> 1
/// [2,4) -> 2
/// [4,8) -> 3
/// ...
inline int intLog2( size_t n )
{
    int l = 0;
    while ( n > 0 )
    {
        ++l;
        n /= 2;
    }
    return l;
}

} //namespace MR
