#include "MRLinesSave.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPch/MRFmt.h"
#include <fstream>

namespace MR
{

namespace LinesSave
{

const IOFilters Filters =
{
    {"MrLines (.mrlines)", "*.mrlines"},
    {"PTS (.pts)", "*.pts"},
    {"Drawing exchange format (.dxf)", "*.dxf"}
};

VoidOrErrStr toMrLines( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrLines( polyline, out, settings );
}

VoidOrErrStr toMrLines( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    MR_TIMER;
    polyline.topology.write( out );

    // write points
    const std::uint32_t type = 3; //3d points
    out.write( (const char*)&type, 4 );
    auto numPoints = (std::uint32_t)( polyline.topology.lastValidVert() + 1 );
    out.write( ( const char* )&numPoints, 4 );
        
    VertCoords buf;
    const auto & xfVerts = transformPoints( polyline.points, polyline.topology.getValidVerts(), settings.xf, buf );
    if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numPoints * sizeof( Vector3f ), settings.progress ) )
        return unexpected( std::string( "Saving canceled" ) );

    if ( !out )
        return unexpected( std::string( "Error saving in MrLines-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toPts( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( polyline, out, settings );
}

VoidOrErrStr toPts( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    float pointsNum{ 0.f };
    auto contours = polyline.contours();
    for ( const auto& contour : contours )
        pointsNum += contour.size();

    int pointIndex = 0;
    for ( const auto& contour : contours )
    {
        out << "BEGIN_Polyline\n";
        for ( auto v : contour )
        {
            auto saveVertex = [&]( auto && p )
            {
                out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
            };
            if ( settings.xf )
                saveVertex( applyDouble( settings.xf, v ) );
            else
                saveVertex( v );
            ++pointIndex;
            if ( settings.progress && !( pointIndex & 0x3FF ) && !settings.progress( float( pointIndex ) / pointsNum ) )
                return unexpected( std::string( "Saving canceled" ) );
        }
        out << "END_Polyline\n";
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PTS-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toDxf( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDxf( polyline, out, settings );
}

VoidOrErrStr toDxf( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    out << "0\nSECTION\n";
    out << "2\nENTITIES\n";

    float pointsNum{ 0.f };
    auto contours = polyline.contours();
    for ( const auto& contour : contours )
        pointsNum += contour.size();

    int pointIndex = 0;
    for ( const auto& contour : contours )
    {
        out << "0\nPOLYLINE\n";
        out << "8\n0\n";
        out << "66\n1\n";
        int flags = 8;
        if ( contour[0] == contour.back() )
            flags += 1;
        out << "70\n" << flags << "\n";
        for ( auto v : contour )
        {
            auto saveVertex = [&]( auto && p )
            {
                out << fmt::format(
                    "0\nVERTEX\n"
                    "8\n0\n"
                    "70\n32\n"
                    "10\n{}\n"
                    "20\n{}\n"
                    "30\n{}\n",
                    p.x, p.y, p.z );
            };
            if ( settings.xf )
                saveVertex( applyDouble( settings.xf, v ) );
            else
                saveVertex( v );
            ++pointIndex;
            if ( settings.progress && !( pointIndex & 0x3FF ) && !settings.progress( float( pointIndex ) / pointsNum ) )
                return unexpected( std::string( "Saving canceled" ) );
        }
        out << "0\nSEQEND\n";
    }

    out << "0\nENDSEC\n";
    out << "0\nEOF\n";

    if ( !out )
        return unexpected( std::string( "Error saving in DXF-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = toMrLines( polyline, file, settings );
    else if ( ext == ".pts" )
        res = toPts( polyline, file, settings );
    else if ( ext == ".dxf" )
        res = toDxf( polyline, file, settings );
    return res;
}

VoidOrErrStr toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension, const SaveSettings & settings )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = toMrLines( polyline, out, settings );
    else if ( ext == ".pts" )
        res = toPts( polyline, out, settings );
    else if ( ext == ".dxf" )
        res = toDxf( polyline, out, settings );
    return res;
}

} //namespace LinesSave

} //namespace MR
