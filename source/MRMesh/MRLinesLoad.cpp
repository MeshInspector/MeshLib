#include "MRLinesLoad.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRStreamOperators.h"
#include <fstream>

namespace MR
{

namespace LinesLoad
{

const IOFilters Filters =
{
    {"MrLines (.mrlines)", "*.mrlines"},
    {"PTS (.pts)",        "*.pts"}
};

Expected<Polyline3, std::string> fromMrLines( const std::filesystem::path & file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromMrLines( in, callback ), file );
}

Expected<Polyline3, std::string> fromMrLines( std::istream & in, ProgressCallback callback )
{
    MR_TIMER

    Polyline3 polyline;
    if ( !polyline.topology.read( in ) )
        return unexpected( std::string( "Error reading topology from lines-file" ) );

    // read points
    std::uint32_t type = 0;
    in.read( (char*)&type, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the type of points from lines-file" ) );
    if ( type != 3 )
        return unexpected( std::string( "Unsupported point type in lines-file" ) );
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of points from lines-file" ) );
    polyline.points.resize( numPoints );
    readByBlocks( in, (char*)polyline.points.data(), polyline.points.size() * sizeof(Vector3f), callback );
    if ( !in )
        return unexpected( std::string( "Error reading  points from lines-file" ) );

    return std::move( polyline );
}

Expected<MR::Polyline3, std::string> fromPts( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPts( in, callback ), file );
}

Expected<MR::Polyline3, std::string> fromPts( std::istream& in, ProgressCallback callback /*= {} */ )
{
    std::string line;
    int pointCount = 0;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    bool isPolylineBlock{ false };

    Polyline3 polyline;
    std::vector<Vector3f> points;
    while ( std::getline( in, line ) )
    {
        line.erase( std::find_if( line.rbegin(), line.rend(), [] ( unsigned char ch )
        {
            return !std::isspace( ch );
        } ).base(), line.end() );
        if ( !isPolylineBlock )
        {
            if ( line != "BEGIN_Polyline" )
                return unexpected( "Not valid .pts format" );
            else
            {
                isPolylineBlock = true;
                continue;
            }
        }
        else if ( line == "END_Polyline" )
        {
            isPolylineBlock = false;
            if ( points.empty() )
                continue;

            polyline.addFromPoints( points.data(), points.size() );
            points.clear();

            continue;
        }

        std::istringstream iss( line );
        Vector3f point;
        if ( !( iss >> point ) )
            return unexpected( "Not valid .pts format" );
        points.push_back( point );
        ++pointCount;

        if ( callback && !( pointCount & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !callback( progress ) )
                return unexpected( std::string( "Loading canceled" ) );
        }
    }
    if ( isPolylineBlock )
        return unexpected( "Not valid .pts format" );

    return std::move( polyline );
}

Expected<Polyline3, std::string> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    Expected<Polyline3, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = fromMrLines( file, callback );
    if ( ext == ".pts" )
        res = fromPts( file, callback );
    return res;
}

Expected<MR::Polyline3, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<Polyline3, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".mrlines" )
        res = fromMrLines( in, callback );
    if ( ext == ".pts" )
        res = fromPts( in, callback );
    return res;
}

} //namespace LinesLoad

} // namespace MR
