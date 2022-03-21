#include "MRLinesSave.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRStringConvert.h"
#include <fstream>

namespace MR
{

namespace LinesSave
{

const IOFilters Filters =
{
    {"MrLines (.mrlines)", "*.mrlines"}
};

tl::expected<void, std::string> toMrLines( const Polyline3& polyline, const std::filesystem::path& file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrLines( polyline, out );
}

tl::expected<void, std::string> toMrLines( const Polyline3& polyline, std::ostream& out )
{
    MR_TIMER;
    polyline.topology.write( out );

    // write points
    const std::uint32_t type = 3; //3d points
    out.write( (const char*)&type, 4 );
    auto numPoints = (std::uint32_t)polyline.points.size();
    out.write( (const char*)&numPoints, 4 );
    out.write( (const char*)polyline.points.data(), polyline.points.size() * sizeof(Vector3f) );

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in MrLines-format" ) );

    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".mrlines" )
        res = toMrLines( polyline, file );
    return res;
}

tl::expected<void, std::string> toAnySupportedFormat( const Polyline3& polyline, std::ostream& out, const std::string& extension )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = toMrLines( polyline, out );
    return res;
}

} //namespace LinesSave

} //namespace MR
