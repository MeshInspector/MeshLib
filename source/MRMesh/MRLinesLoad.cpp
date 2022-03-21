#include "MRLinesLoad.h"
#include "MRPolyline3.h"
#include "MRTimer.h"
#include "MRStringConvert.h"
#include <fstream>

namespace MR
{

namespace LinesLoad
{

const IOFilters Filters =
{
    {"MrLines (.mrlines)", "*.mrlines"}
};

tl::expected<Polyline3, std::string> fromMrLines( const std::filesystem::path & file )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromMrLines( in );
}

tl::expected<Polyline3, std::string> fromMrLines( std::istream & in )
{
    MR_TIMER

    Polyline3 polyline;
    if ( !polyline.topology.read( in ) )
        return tl::make_unexpected( std::string( "Error reading topology from lines-file" ) );

    // read points
    std::uint32_t type = 0;
    in.read( (char*)&type, 4 );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading the type of points from lines-file" ) );
    if ( type != 3 )
        return tl::make_unexpected( std::string( "Unsupported point type in lines-file" ) );
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading the number of points from lines-file" ) );
    polyline.points.resize( numPoints );
    in.read( (char*)polyline.points.data(), polyline.points.size() * sizeof(Vector3f) );
    if ( !in )
        return tl::make_unexpected( std::string( "Error reading  points from lines-file" ) );

    return std::move( polyline );
}

tl::expected<Polyline3, std::string> fromAnySupportedFormat( const std::filesystem::path& file )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<Polyline3, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".mrlines" )
        res = fromMrLines( file );
    return res;
}

tl::expected<MR::Polyline3, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<Polyline3, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = fromMrLines( in );
    return res;
}

} //namespace LinesLoad

} // namespace MR
