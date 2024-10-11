#include "MRLinesSave.h"
#include "MRIOFormatsRegistry.h"
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

Expected<void> toMrLines( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrLines( polyline, out, settings );
}

Expected<void> toMrLines( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
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

Expected<void> toPts( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( polyline, out, settings );
}

Expected<void> toPts( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
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

Expected<void> toDxf( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDxf( polyline, out, settings );
}

Expected<void> toDxf( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
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

Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getLinesSaver( ext );
    if ( !saver.fileSave )
        return unexpected( std::string( "unsupported file extension" ) );

    return saver.fileSave( polyline, file, settings );
}

Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::string& extension, std::ostream& out, const SaveSettings & settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getLinesSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    return saver.streamSave( polyline, out, settings );
}

MR_ADD_LINES_SAVER_WITH_PRIORITY( IOFilter( "MrLines (.mrlines)", "*.mrlines" ), toMrLines, -1 )
MR_ADD_LINES_SAVER( IOFilter( "PTS (.pts)", "*.pts" ), toPts )
MR_ADD_LINES_SAVER( IOFilter( "Drawing exchange format (.dxf)", "*.dxf" ), toDxf )

} //namespace LinesSave

} //namespace MR
