#include "MRTelemetry.h"
#include "MRString.h"

namespace MR
{

void telemetryStlHead( std::string_view s )
{
    s = trimLeft( s );
    s = trimRight( s );

    auto n = s.find( "COLOR=" );
    if ( n != std::string_view::npos && n + 6 + 4 <= s.size() )
    {
        n += 6;
        for ( int i = 0; i < 4; ++i, ++n )
            s[n] = '_';
    }

    TelemetrySignal( "STL head " + std::string( s ) );
}

} //namespace MR
