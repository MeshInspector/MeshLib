#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "MRPly.h"
#include "MRColor.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPointCloud.h"
#include "MRIOParsing.h"
#include "MRParallelFor.h"
#include "MRComputeBoundingBox.h"
#include "MRBitSetParallelFor.h"

#include <fstream>

namespace MR::PointsLoad
{

Expected<PointCloud> fromText( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromText( in, settings ), file );
}

Expected<PointCloud> fromText( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;

    auto buf = readCharBuffer( in );
    if ( !buf )
        return unexpected( std::move( buf.error() ) );

    if ( !reportProgress( settings.callback, 0.50f ) )
        return unexpectedOperationCanceled();

    const auto newlines = splitByLines( buf->data(), buf->size() );
    const auto lineCount = newlines.size() - 1;

    if ( !reportProgress( settings.callback, 0.60f ) )
        return unexpectedOperationCanceled();

    PointCloud cloud;
    cloud.points.resizeNoInit( lineCount );
    cloud.validPoints.resize( lineCount, false );

    // detect normals and colors
    constexpr Vector3d cInvalidNormal( 0.f, 0.f, 0.f );
    constexpr Color cInvalidColor( 0, 0, 0, 0 );
    Vector3d firstPoint;
    auto hasNormals = false;
    auto hasColors = false;
    for ( auto i = 0; i < lineCount; ++i )
    {
        const std::string_view line( buf->data() + newlines[i], newlines[i + 1] - newlines[i + 0] );
        if ( line.empty() || line.starts_with( '#' ) || line.starts_with( ';' ) )
            continue;

        auto normal = cInvalidNormal;
        auto color = cInvalidColor;
        auto result = parseTextCoordinate( line, firstPoint, &normal, &color );
        if ( !result )
            return unexpected( std::move( result.error() ) );

        if ( settings.outXf )
            *settings.outXf = AffineXf3f::translation( Vector3f( firstPoint ) );

        if ( normal != cInvalidNormal )
        {
            hasNormals = true;
            cloud.normals.resizeNoInit( lineCount );
        }
        if ( settings.colors && color != cInvalidColor )
        {
            hasColors = true;
            settings.colors->resizeNoInit( lineCount );
        }

        break;
    }

    std::string parseError;
    tbb::task_group_context ctx;
    const auto keepGoing = BitSetParallelForAll( cloud.validPoints, [&] ( VertId v )
    {
        const std::string_view line( buf->data() + newlines[v], newlines[v + 1] - newlines[v + 0] );
        if ( line.empty() || line.starts_with( '#' ) || line.starts_with( ';' ) )
            return;

        Vector3d point( noInit );
        Vector3d normal( noInit );
        Color color( noInit );
        auto result = parseTextCoordinate( line, point, hasNormals ? &normal : nullptr, hasColors ? &color : nullptr );
        if ( !result )
        {
            if ( ctx.cancel_group_execution() )
                parseError = std::move( result.error() );
            return;
        }

        cloud.points[v] = Vector3f( settings.outXf ? point - firstPoint : point );
        cloud.validPoints.set( v, true );
        if ( hasNormals )
            cloud.normals[v] = Vector3f( normal );
        if ( hasColors )
            ( *settings.colors )[v] = color;
    }, subprogress( settings.callback, 0.60f, 1.00f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();
    if ( !parseError.empty() )
        return unexpected( std::move( parseError ) );

    return cloud;
}

Expected<PointCloud> fromPts( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPts( in, settings ), file );
}

Expected<PointCloud> fromPts( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;
    auto startPos = in.tellg();
    std::string numPointsLine;
    if ( !std::getline( in, numPointsLine ) )
        return unexpected( "Cannot read header line" );

    Vector3f testFirstLine;
    if ( parseTextCoordinate( numPointsLine, testFirstLine ).has_value() )
    {
        // asc-like pts file
        in.clear();
        in.seekg( startPos );
        return fromText( in, settings );
    }

    auto numPoints = std::atoll( numPointsLine.c_str() );
    if ( numPoints == 0 )
        return unexpected( "Empty pts file" );

    auto dataExp = readCharBuffer( in );
    if ( !dataExp.has_value() )
        return unexpected( dataExp.error() );

    if ( settings.callback && !settings.callback( 0.25f ) )
        return unexpectedOperationCanceled();

    const auto& data = *dataExp;
    auto lineOffsets = splitByLines( data.data(), data.size() );

    int firstLine = 1;
    Vector3d firstLineCoord;
    Color firstLineColor;
    std::string_view shitLine( data.data() + lineOffsets[firstLine], lineOffsets[firstLine + 1] - lineOffsets[firstLine] );
    auto shiftLineRes = parsePtsCoordinate( shitLine, firstLineCoord, firstLineColor );
    if ( !shiftLineRes.has_value() )
        return unexpected( shiftLineRes.error() );

    if ( settings.outXf )
        *settings.outXf = AffineXf3f::translation( Vector3f( firstLineCoord ) );

    if ( settings.colors )
        settings.colors->resize( lineOffsets.size() - firstLine - 1 );

    PointCloud pc;
    pc.points.resize( lineOffsets.size() - firstLine - 1 );

    std::string parseError;
    tbb::task_group_context ctx;
    auto keepGoing = ParallelFor( pc.points, [&] ( size_t i )
    {
        std::string_view line( data.data() + lineOffsets[firstLine + i], lineOffsets[firstLine + i + 1] - lineOffsets[firstLine + i] );
        Vector3d tempDoubleCoord;
        Color tempColor;
        auto parseRes = parsePtsCoordinate( line, tempDoubleCoord, tempColor );
        if ( !parseRes.has_value() && ctx.cancel_group_execution() )
            parseError = std::move( parseRes.error() );

        pc.points[VertId( i )] = Vector3f( tempDoubleCoord - firstLineCoord );
        if ( settings.colors )
            ( *settings.colors )[VertId( i )] = tempColor;
    }, subprogress( settings.callback, 0.25f, 1.0f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    if ( !parseError.empty() )
        return unexpected( parseError );

    pc.validPoints.resize( pc.points.size(), true );
    return pc;
}

Expected<PointCloud> fromPly( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, settings ), file );
}

Expected<PointCloud> fromPly( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;

    PointCloud res;
    PlyLoadParams params =
    {
        .colors = settings.colors,
        .normals = &res.normals,
        .callback = settings.callback
    };
    auto maybePoints = loadPly( in, params );
    if ( !maybePoints )
        return unexpected( std::move( maybePoints.error() ) );

    res.points = std::move( *maybePoints );
    res.validPoints.resize( res.points.size(), true );
    return res;
}

Expected<PointCloud> fromObj( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromObj( in, settings ), file );
}

Expected<PointCloud> fromObj( std::istream& in, const PointsLoadSettings& settings )
{
    PointCloud cloud;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    for ( int i = 0;; ++i )
    {
        if ( !in )
            return unexpected( std::string( "OBJ-format read error" ) );
        char ch = 0;
        in >> ch;
        if ( in.eof() )
            break;
        if ( ch == 'v' )
        {
            float x, y, z;
            in >> x >> y >> z;
            cloud.points.emplace_back( x, y, z );
        }
        else
        {
            // skip unknown line
            std::string str;
            std::getline( in, str );
        }

        if ( settings.callback && !( i & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / float( streamSize );
            if ( !settings.callback( progress ) )
                return unexpectedOperationCanceled();
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return cloud;
}

Expected<PointCloud> fromDxf( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromDxf( in, settings ), file );
}

Expected<PointCloud> fromDxf( std::istream& in, const PointsLoadSettings& settings )
{
    PointCloud cloud;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    std::string str;
    std::getline( in, str );

    int code{};
    if ( !parseSingleNumber<int>( str, code ) )
        return unexpected( "File is corrupted" );

    bool isPointFound = false;

    for ( int i = 0; !in.eof(); ++i )
    {
        if ( i % 1024 == 0 && !reportProgress( settings.callback, float( in.tellg() - posStart ) / float( streamSize ) ) )
            return unexpectedOperationCanceled();

        std::getline( in, str );

        if ( str == "POINT" )
        {
            cloud.points.emplace_back();
            isPointFound = true;
        }

        if ( isPointFound )
        {
            const int vIdx = code % 10;
            const int cIdx = code / 10 - 1;
            if ( vIdx == 0 && cIdx >= 0 && cIdx < 3 )
            {
                if ( !parseSingleNumber<float>( str, cloud.points.back()[cIdx] ) )
                    return unexpected( "File is corrupted" );
            }
        }

        std::getline( in, str );
        if ( str.empty() )
            continue;

        if ( !parseSingleNumber<int>( str, code ) )
            return unexpected( "File is corrupted" );

        if ( code == 0 )
            isPointFound = false;
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( cloud.points.empty() )
        return unexpected( "No points are found " );

    cloud.validPoints.resize( cloud.points.size(), true );
    return cloud;
}

Expected<PointCloud> fromAnySupportedFormat( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto loader = getPointsLoader( ext );
    if ( !loader.fileLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.fileLoad( file, settings );
}

Expected<PointCloud> fromAnySupportedFormat( std::istream& in, const std::string& extension, const PointsLoadSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = getPointsLoader( ext );
    if ( !loader.streamLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.streamLoad( in, settings );
}

MR_ADD_POINTS_LOADER( IOFilter( "ASC (.asc)",        "*.asc" ), fromText )
MR_ADD_POINTS_LOADER( IOFilter( "CSV (.csv)",        "*.csv" ), fromText )
MR_ADD_POINTS_LOADER( IOFilter( "XYZ (.xyz)",        "*.xyz" ), fromText )
MR_ADD_POINTS_LOADER( IOFilter( "XYZ (.xyzn)",       "*.xyzn" ), fromText )
MR_ADD_POINTS_LOADER( IOFilter( "OBJ (.obj)",        "*.obj" ), fromObj )
MR_ADD_POINTS_LOADER( IOFilter( "PLY (.ply)",        "*.ply" ), fromPly )
MR_ADD_POINTS_LOADER( IOFilter( "LIDAR scanner (.pts)", "*.pts" ), fromPts )
MR_ADD_POINTS_LOADER( IOFilter( "DXF (.dxf)",        "*.dxf" ), fromDxf )

} // namespace MR::PointsLoad
