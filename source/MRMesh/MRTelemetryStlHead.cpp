#include "MRTelemetry.h"

namespace MR
{

void telemetryStlHead( std::string s )
{
    while ( !s.empty() && ( s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n' ) )
        s.pop_back();
    while ( !s.empty() && ( s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n' ) )
        s = s.substr( 1 );

    // replace specific color with underscores
    const char COLOR[] = "COLOR=";
    static_assert( sizeof( COLOR ) == 7 );
    auto n = s.find( COLOR );
    if ( n != std::string::npos && n + sizeof( COLOR )-1 + 4 <= s.size() )
    {
        n += sizeof( COLOR )-1;
        for ( int i = 0; i < 4; ++i, ++n )
            s[n] = '_';
    }

    // replace specific material colors with underscores
    const char MATERIAL[] = "MATERIAL=";
    n = s.find( COLOR );
    if ( n != std::string::npos && n + sizeof( MATERIAL )-1 + 12 <= s.size() )
    {
        n += sizeof( MATERIAL )-1;
        for ( int i = 0; i < 12; ++i, ++n )
            s[n] = '_';
    }

    TelemetrySignal( "STL head " + s );
}

} //namespace MR
