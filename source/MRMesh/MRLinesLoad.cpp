#include "MRLinesLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOParsing.h"
#include "MRPolyline.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRStreamOperators.h"
#include "MRPly.h"
#include "MRTimer.h"
#include <fstream>

namespace MR
{

namespace LinesLoad
{

Expected<Polyline3> fromMrLines( const std::filesystem::path & file, const LinesLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromMrLines( in, settings ), file );
}

Expected<Polyline3> fromMrLines( std::istream & in, const LinesLoadSettings& settings )
{
    MR_TIMER;

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
    readByBlocks( in, (char*)polyline.points.data(), polyline.points.size() * sizeof(Vector3f), settings.callback );
    if ( !in )
        return unexpected( std::string( "Error reading  points from lines-file" ) );

    return polyline;
}

Expected<MR::Polyline3> fromPts( const std::filesystem::path& file, const LinesLoadSettings& settings /*= {} */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPts( in, settings ), file );
}

Expected<MR::Polyline3> fromPts( std::istream& in, const LinesLoadSettings& settings /*= {} */ )
{
    std::string line;
    int pointCount = 0;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

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

        if ( settings.callback && !( pointCount & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / float( streamSize );
            if ( !settings.callback( progress ) )
                return unexpectedOperationCanceled();
        }
    }
    if ( isPolylineBlock )
        return unexpected( "Not valid .pts format" );

    return polyline;
}

Expected<Polyline3> fromPly( const std::filesystem::path& file, const LinesLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, settings ), file );
}

Expected<Polyline3> fromPly( std::istream& in, const LinesLoadSettings& settings )
{
    MR_TIMER;

    std::optional<Edges> edges;
    PlyLoadParams params =
    {
        .edges = &edges,
        .colors = settings.colors,
        // suppose that reading is 10% of progress and building polyline is 90% of progress
        .callback = subprogress( settings.callback, 0.0f, 0.1f )
    };
    auto maybePoints = loadPly( in, params );
    if ( !maybePoints )
        return unexpected( std::move( maybePoints.error() ) );

    Polyline3 res;
    res.points = std::move( *maybePoints );

    if ( edges )
    {
        res.topology.vertResize( res.points.size() );
        res.topology.makeEdges( *edges );
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<Polyline3> fromAnySupportedFormat( const std::filesystem::path& file, const LinesLoadSettings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto loader = getLinesLoader( ext );
    if ( !loader.fileLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.fileLoad( file, settings );
}

Expected<MR::Polyline3> fromAnySupportedFormat( std::istream& in, const std::string& extension, const LinesLoadSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = getLinesLoader( ext );
    if ( !loader.streamLoad )
        return unexpected( std::string( "unsupported stream extension" ) );

    return loader.streamLoad( in, settings );
}

MR_ADD_LINES_LOADER_WITH_PRIORITY( IOFilter( "MrLines (.mrlines)", "*.mrlines" ), fromMrLines, -1 )
MR_ADD_LINES_LOADER( IOFilter( "PTS (.pts)",         "*.pts" ),     fromPts )
MR_ADD_LINES_LOADER( IOFilter( "PLY (.ply)",         "*.ply" ),     fromPly )

} //namespace LinesLoad

} // namespace MR
