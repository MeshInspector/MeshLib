#include "MRTelemetry.h"

namespace MR
{

namespace
{

bool digitsAt( const std::string& s, size_t pos, size_t cnt )
{
    if ( pos + cnt > s.size() )
        return false;
    for ( size_t i = 0; i < cnt; ++i )
        if ( s[pos + i] < '0' || s[pos + i] > '9' )
            return false;
    return true;
}

} //anonymous namespace


/// removes white spaces, meaningless or case-specific information from a comment line, then calls telemetry signal
void telemetryStlHead( const char* prefix, std::string s )
{
    while ( !s.empty() && ( s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n' ) )
        s.pop_back();
    while ( !s.empty() && ( s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n' ) )
        s = s.substr( 1 );

    // replace specific material colors with underscores
    const char MATERIAL[] = "MATERIAL=";
    auto n = s.find( MATERIAL );
    const bool materialFound = n != std::string::npos;
    if ( materialFound )
    {
        n += sizeof( MATERIAL )-1;
        for ( int i = 0; i < 12 && n < s.size(); ++i, ++n )
            s[n] = '_';
    }

    const char COLOR[] = "COLOR=";
    static_assert( sizeof( COLOR ) == 7 );
    if ( !materialFound && s.starts_with( COLOR ) )
    {
        // the size can be arbitrary with some not-printable characters after
        s.resize( sizeof( COLOR ) - 1 );
    }
    else
    {
        // replace specific color with underscores
        n = s.find( COLOR );
        if ( n != std::string::npos )
        {
            n += sizeof( COLOR )-1;
            for ( int i = 0; i < 4 && n < s.size(); ++i, ++n )
                s[n] = '_';
        }
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

    // e.g. "$objname"
    if ( s.starts_with( "$" ) )
        s = "$objname";

    // e.g. "\"objname\""
    if ( s.size() >= 2 && s.front() == '"' && s.back() == '"' )
        s = "\"objname\"";

    // e.g. "objname.stl"
    if ( s.ends_with( ".stl" ) )
        s = "objname.stl";

    // e.g. "objname.stl (spaces) COLOR=____"
    if ( s.size() == 80 && s.substr( 70, sizeof( COLOR ) - 1 ) == COLOR )
    {
        n = s.find( ".stl " );
        if ( n != std::string::npos )
            s = "objname.stl COLOR=____";
    }

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

    // e.g. 'mo_file_id: 447338 file_source: 2'
    const char MO_FILE_ID[] = "mo_file_id: ";
    if ( s.starts_with( MO_FILE_ID ) )
        s.resize( sizeof( MO_FILE_ID ) - 2 );

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

    // e.g. 'STL EXPORTED BY IDEAMAKER. 14-02-2026 22:21:37'
    const char IDEAMAKER[] = "STL EXPORTED BY IDEAMAKER. ";
    if ( s.starts_with( IDEAMAKER ) )
        s.resize( sizeof( IDEAMAKER ) - 2 );

    // e.g. 'SOLID RELIEF MANIFOLD - 20260211-235703'
    const char RELIEF[] = "SOLID RELIEF MANIFOLD - ";
    if ( s.starts_with( RELIEF ) )
        s.resize( sizeof( RELIEF ) - 2 );

    // e.g. 'Created by stlwrite.m 29-Apr-2022 07:10:05'
    const char STLWRITE_M[] = "Created by stlwrite.m ";
    if ( s.starts_with( STLWRITE_M ) )
        s.resize( sizeof( STLWRITE_M ) - 2 );

    // e.g. '# STL binary facet file Clip.stl, v. 18.0, made 18:03, Jan 07, 2018'
    const char FACET_FILE[] = "# STL binary facet file ";
    if ( s.starts_with( FACET_FILE ) )
        s.resize( sizeof( FACET_FILE ) - 2 );

    // e.g. 'Created by surf2stl.m 23-Feb-2026 13:24:15'
    const char SURf2STL[] = "Created by surf2stl.m ";
    if ( s.starts_with( SURf2STL ) )
        s.resize( sizeof( SURf2STL ) - 1 );

    // e.g. 'Guardian AI Hub Bottom Shell'
    const char GuardianAI[] = "Guardian AI Hub ";
    if ( s.starts_with( GuardianAI ) )
        s.resize( sizeof( GuardianAI ) - 1 );

    // e.g. 'numpy-stl (3.0.0) 2026-01-05 14:46:07.404027 tmphpyx9npt.stl'
    const char NUMPY[] = "numpy-stl (";
    if ( s.starts_with( NUMPY ) )
        s = s.substr( 0, s.find_first_of( ' ', sizeof( NUMPY ) ) ); // till the space after version

    // e.g. '[7uy4DjmtrmckCEK7ZCbGxK-55bc] Generated by CADflow.ai'
    const char CADFLOW_AI[] = "Generated by CADflow.ai";
    if ( s.ends_with( CADFLOW_AI ) )
        s = CADFLOW_AI;

    // e.g. 'BlueSkyPlan 5.0.29.1 03/06/2026 09:31:47 UTC' - keep software and version, drop the export date and time
    if ( s.starts_with( "BlueSkyPlan " ) )
    {
        const auto slash = s.find( '/' ); // first slash belongs to the dd/mm/yyyy date (version uses dots)
        if ( slash != std::string::npos )
        {
            const auto sp = s.rfind( ' ', slash );
            if ( sp != std::string::npos )
                s.resize( sp );
        }
    }

    // e.g. multi-line 'VXelements_Binary_STL\n1\n1\n20260525\n14416' with per-file date and key numbers
    if ( s.starts_with( "VXelements_Binary_STL" ) )
        s = "VXelements_Binary_STL";

    // e.g. multi-line 'Binary STL file generated by VXelements\nDate:20170920\nKey:57933'
    const char VXELEMENTS[] = "Binary STL file generated by VXelements";
    if ( s.starts_with( VXELEMENTS ) )
        s.resize( sizeof( VXELEMENTS ) - 1 );

    // e.g. 'Scaniverse 2026-06-12 181333' - trailing capture date and time
    const char SCANIVERSE[] = "Scaniverse ";
    if ( s.starts_with( SCANIVERSE ) )
        s.resize( sizeof( SCANIVERSE ) - 2 );

    // e.g. 'Untitled-6E4A1B39'
    const char UNTITLED[] = "Untitled-";
    if ( s.starts_with( UNTITLED ) )
        s.resize( sizeof( UNTITLED ) - 2 );

    // e.g. 'Exported from UVtools v5.1.6 @ 2026-02-14 22:21:37Z' - keep the version, drop the export time
    const char UVTOOLS[] = "Exported from UVtools ";
    if ( s.starts_with( UVTOOLS ) )
    {
        const auto at = s.find( " @ " );
        if ( at != std::string::npos )
            s.resize( at );
    }

    // e.g. 'exocad GmbH 2026 - DentalCAD' - the release year varies per installation
    const char EXOCAD_DE[] = "exocad GmbH ";
    const char EXOCAD_US[] = "exocad North America ";
    auto dropYear = [&s]( const char* pref, size_t len )
    {
        if ( s.starts_with( pref ) && digitsAt( s, len, 4 ) && len + 4 < s.size() && s[len + 4] == ' ' )
            s.erase( len, 5 );
    };
    dropYear( EXOCAD_DE, sizeof( EXOCAD_DE ) - 1 );
    dropYear( EXOCAD_US, sizeof( EXOCAD_US ) - 1 );

    // e.g. 'TopoMiller 2026-01-02 15:59:18' - trailing export date, with optional time
    for ( size_t i = 0; i + 10 <= s.size(); ++i )
    {
        if ( !digitsAt( s, i, 4 ) || ( s[i + 4] != '-' && s[i + 4] != '/' ) ||
             !digitsAt( s, i + 5, 2 ) || s[i + 7] != s[i + 4] || !digitsAt( s, i + 8, 2 ) )
            continue;
        size_t e = i + 10;
        if ( e < s.size() && ( s[e] == ' ' || s[e] == 'T' ) )
        {
            size_t t = e + 1;
            while ( t < s.size() && ( digitsAt( s, t, 1 ) || s[t] == ':' || s[t] == '.' ) )
                ++t;
            if ( t < s.size() && s[t] == 'Z' )
                ++t;
            if ( t > e + 1 )
                e = t;
        }
        if ( e != s.size() )
            continue; // the date is not trailing, it is a part of the name
        while ( i > 0 && ( s[i - 1] == ' ' || s[i - 1] == '-' || s[i - 1] == ',' || s[i - 1] == ':' ) )
            --i;
        s.resize( i );
        break;
    }

    // e.g. 'RACK_INLET_COLD_017', 'A - TO.Ankylos X_Geo-65' - per-file sequence numbers;
    // a space separator is not accepted, it would eat model numbers like 'CS 3600'
    if ( const auto sep = s.find_last_not_of( "0123456789" );
         sep != std::string::npos && sep > 0 && sep + 1 < s.size() && s.size() - sep <= 7 &&
         ( s[sep] == '_' || s[sep] == '-' ) )
        s.resize( sep );

    TelemetrySignal( prefix + s );
}

} //namespace MR
