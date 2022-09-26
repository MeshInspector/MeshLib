#include "MRLinesSave.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
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

tl::expected<void, std::string> toMrLines( const Polyline3& polyline, const std::filesystem::path& file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrLines( polyline, out, callback );
}

tl::expected<void, std::string> toMrLines( const Polyline3& polyline, std::ostream& out, ProgressCallback callback )
{
    MR_TIMER;
    polyline.topology.write( out );

    // write points
    const std::uint32_t type = 3; //3d points
    out.write( (const char*)&type, 4 );
    auto numPoints = (std::uint32_t)polyline.points.size();
    out.write( ( const char* )&numPoints, 4 );
        
    const bool cancel = !MR::writeByBlocks( out, (const char*) polyline.points.data(), polyline.points.size() * sizeof( Vector3f ), callback );
    if ( cancel )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in MrLines-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> toPts( const Polyline3& polyline, const std::filesystem::path& file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( polyline, out, callback );
}

tl::expected<void, std::string> toPts( const Polyline3& polyline, std::ostream& out, ProgressCallback callback )
{
    float pointsNum{ 0.f };
    auto contours = polyline.contours();
    for ( const auto& contour : contours )
        pointsNum += contour.size();

    int pointIndex = 0;
    for ( const auto& contour : contours )
    {
        out << "BEGIN_Polyline\n";
        for ( auto p : contour )
        {
            out << p << "\n";
            ++pointIndex;
            if ( callback && !( pointIndex & 0x3FF ) && !callback( float( pointIndex ) / pointsNum ) )
                return tl::make_unexpected( std::string( "Saving canceled" ) );
        }
        out << "END_Polyline\n";
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PTS-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> toDxf( const Polyline3& polyline, const std::filesystem::path& file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDxf( polyline, out, callback );
}

tl::expected<void, std::string> toDxf( const Polyline3& polyline, std::ostream& out, ProgressCallback callback )
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
        for ( auto p : contour )
        {
            out << "0\nVERTEX\n";
            out << "8\n0\n";
            out << "70\n32\n";
            out << "10\n" << p.x << "\n";
            out << "20\n" << p.y << "\n";
            out << "30\n" << p.z << "\n";
            ++pointIndex;
            if ( callback && !( pointIndex & 0x3FF ) && !callback( float( pointIndex ) / pointsNum ) )
                return tl::make_unexpected( std::string( "Saving canceled" ) );
        }
        out << "0\nSEQEND\n";
    }

    out << "0\nENDSEC\n";
    out << "0\nEOF\n";

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in DXF-format" ) );

    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, ProgressCallback callback )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = toMrLines( polyline, file, callback );
    if ( ext == ".pts" )
        res = toPts( polyline, file, callback );
    if ( ext == ".dxf" )
        res = toDxf( polyline, file, callback );
    return res;
}

tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension, ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = toMrLines( polyline, out, callback );
    if ( ext == ".pts" )
        res = toPts( polyline, out, callback );
    if ( ext == ".dxf" )
        res = toDxf( polyline, out, callback );
    return res;
}

} //namespace LinesSave

} //namespace MR
