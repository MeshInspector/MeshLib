#include "MRTelemetry.h"

namespace MR
{

/// removes white spaces, meaningless or case-specific information from a comment line, then calls telemetry signal
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

    // e.g. "solid objname"
    const char SOLID[] = "solid ";
    if ( s.starts_with( SOLID ) )
        s.resize( sizeof( SOLID ) - 2 );

    // e.g. "stlbn objname"
    const char STLBN[] = "stlbn ";
    if ( s.starts_with( STLBN ) )
        s.resize( sizeof( STLBN ) - 2 );

    // e.g. "STLEXP objname"
    const char STLEXP[] = "STLEXP ";
    if ( s.starts_with( STLEXP ) )
        s.resize( sizeof( STLEXP ) - 2 );

    // e.g. "objname.stl"
    if ( s.ends_with( ".stl" ) )
        s = "objname.stl";

    // e.g. 'SketchUp STL tmpHEPDHM'
    const char SKETCHUP[] = "SketchUp STL ";
    if ( s.starts_with( SKETCHUP ) )
        s.resize( sizeof( SKETCHUP ) - 2 );

    // e.g. '3Design CAD STL : part0'
    const char THREEDESIGN[] = "3Design CAD STL :";
    if ( s.starts_with( THREEDESIGN ) )
        s.resize( sizeof( THREEDESIGN ) - 3 );

    // e.g. 'MW 1.0 1012069 US'
    const char MW10[] = "MW 1.0 ";
    if ( s.starts_with( MW10 ) )
        s.resize( sizeof( MW10 ) - 2 );

    // e.g. 'ML US IM3Dv2 176841983643314567'
    const char MLUS[] = "ML US ";
    if ( s.starts_with( MLUS ) )
        s.resize( sizeof( MLUS ) - 2 );

    // e.g. 'flashforge stl export: %120260113 04:27:48'
    const char FLASHFORGE[] = "flashforge stl export: ";
    if ( s.starts_with( FLASHFORGE ) )
        s.resize( sizeof( FLASHFORGE ) - 2 );

    // e.g. 'Uranium STLWriter Wed 07 Jan 2026 22:23:13'
    const char URANIUM[] = "Uranium STLWriter ";
    if ( s.starts_with( URANIUM ) )
        s.resize( sizeof( URANIUM ) - 2 );

    // e.g. CURA BINARY STL EXPORT. Mon 05 Jan 2026 22:22:20
    const char CURA[] = "CURA BINARY STL EXPORT. ";
    if ( s.starts_with( CURA ) )
        s.resize( sizeof( CURA ) - 2 );

    // e.g. TopoMiller Streaming STL 2026-01-02T15:59:18.983Z
    const char TOPOMILLER[] = "TopoMiller Streaming STL ";
    if ( s.starts_with( TOPOMILLER ) )
        s.resize( sizeof( TOPOMILLER ) - 2 );

    // e.g. "numpy-stl (3.0.0) 2026-01-05 14:46:07.404027 tmphpyx9npt.stl"
    const char NUMPY[] = "numpy-stl (";
    if ( s.starts_with( NUMPY ) )
        s = s.substr( 0, s.find_first_of( ' ', sizeof( NUMPY ) ) ); // till the space after version

    TelemetrySignal( "STL head " + s );
}

} //namespace MR
