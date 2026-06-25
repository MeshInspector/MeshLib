#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "MRPly.h"
#include "MRColor.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRPointCloud.h"
#include "MRIOParsing.h"
#include "MRParallelFor.h"
#include "MRComputeBoundingBox.h"
#include "MRBitSetParallelFor.h"
#include "MRTelemetry.h"
#include "MRDirectory.h"
#include "MRAffineXf3.h"
#include "MRPch/MRSpdlog.h"

#include <fstream>
#include <set>
#include <map>
#include <charconv>
#include <optional>

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

    auto bufData = buf->data();
    auto bufSize = buf->size();
    if ( hasBom( { bufData, bufSize } ) )
    {
        bufData += 3;
        bufSize -= 3;
    }

    if ( !reportProgress( settings.callback, 0.50f ) )
        return unexpectedOperationCanceled();

    const auto newlines = splitByLines( bufData, bufSize );
    const auto lineCount = newlines.size() - 1;

    if ( !reportProgress( settings.callback, 0.60f ) )
        return unexpectedOperationCanceled();

    PointCloud cloud;
    cloud.points.resizeNoInit( lineCount );
    cloud.validPoints.resize( lineCount, false );

    // CSV format specification (RFC 4180) doesn't support comments.
    // However, there's several unofficial conventions to mark lines as comments rather than data records.
    // Some examples:
    // https://stackoverflow.com/questions/1961006/can-a-csv-file-have-a-comment
    // https://giftoolscookbook.readthedocs.io/en/latest/content/fileFormats/CSVfile.html
    static const std::set<char> cCommentChars { '#', ';', '/', '!', '%' };

    // detect normals and colors
    constexpr Vector3d cInvalidNormal( 0.f, 0.f, 0.f );
    constexpr Color cInvalidColor( 0, 0, 0, 0 );
    Vector3d firstPoint;
    auto hasNormals = false;
    auto hasColors = false;
    std::string_view header;
    for ( auto i = 0; i < lineCount; ++i )
    {
        const std::string_view line( bufData + newlines[i], newlines[i + 1] - newlines[i + 0] );
        if ( line.empty() || cCommentChars.contains( line[0] ) )
            continue;

        auto normal = cInvalidNormal;
        auto color = cInvalidColor;
        auto result = parseTextCoordinate( line, firstPoint, &normal, &color );
        if ( !result )
        {
            if ( header.empty() )
                header = line;
            continue;
        }

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

    BitSet parseErrorLines( lineCount, false );
    const auto keepGoing = BitSetParallelForAll( cloud.validPoints, [&] ( VertId v )
    {
        const std::string_view line( bufData + newlines[v], newlines[v + 1] - newlines[v + 0] );
        if ( line.empty() || cCommentChars.contains( line[0] ) )
            return;

        Vector3d point( noInit );
        Vector3d normal( noInit );
        Color color( noInit );
        auto result = parseTextCoordinate( line, point, hasNormals ? &normal : nullptr, hasColors ? &color : nullptr );
        if ( !result )
        {
            // ignore the parse error for the first line, assuming it is a header
            if ( line.data() != header.data() )
                parseErrorLines.set( v.get() );
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
    if ( cloud.validPoints.none() )
        return unexpected( "Could not read any points" );

    if ( parseErrorLines.any() )
    {
        if ( parseErrorLines.count() < 10 )
        {
            std::ostringstream oss;
            bool empty = true;
            for ( auto line : parseErrorLines )
            {
                if ( !empty )
                    oss << ", ";
                oss << line;
                empty = false;
            }
            spdlog::info( "Failed to parse lines: {}", oss.str() );
        }
        else
        {
            spdlog::info( "Failed to parse {} lines", parseErrorLines.count() );
        }
    }

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

    if ( hasBom( numPointsLine ) )
        numPointsLine = numPointsLine.substr( 3 );

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

    // detect normals and colors
    constexpr Vector3d cInvalidNormal( 0.f, 0.f, 0.f );
    constexpr Color cInvalidColor( 0, 0, 0, 0 );
    int firstLine = 1;
    Vector3d firstLineCoord;
    Vector3d firstLineNormal = cInvalidNormal;
    Color firstLineColor = cInvalidColor;
    auto hasNormals = false;
    auto hasColors = false;
    std::string_view shiftLine( data.data() + lineOffsets[firstLine], lineOffsets[firstLine + 1] - lineOffsets[firstLine] );
    auto shiftLineRes = parsePtsCoordinate( shiftLine, firstLineCoord, &firstLineColor, &firstLineNormal );
    if ( !shiftLineRes.has_value() )
        return unexpected( shiftLineRes.error() );

    if ( settings.outXf )
        *settings.outXf = AffineXf3f::translation( Vector3f( firstLineCoord ) );

    if ( settings.colors && firstLineColor != cInvalidColor )
    {
        hasColors = true;
        settings.colors->resizeNoInit( lineOffsets.size() - firstLine - 1 );
    }

    PointCloud pc;
    pc.points.resizeNoInit( lineOffsets.size() - firstLine - 1 );

    if ( firstLineNormal != cInvalidNormal )
    {
        hasNormals = true;
        pc.normals.resizeNoInit( lineOffsets.size() - firstLine - 1 );
    }

    std::string parseError;
    tbb::task_group_context ctx;
    auto keepGoing = ParallelFor( pc.points, [&] ( size_t i )
    {
        std::string_view line( data.data() + lineOffsets[firstLine + i], lineOffsets[firstLine + i + 1] - lineOffsets[firstLine + i] );
        Vector3d tempDoubleCoord;
        Vector3d tempDoubleNormal;
        Color tempColor;
        auto parseRes = parsePtsCoordinate( line, tempDoubleCoord, hasColors ? &tempColor : nullptr, hasNormals ? &tempDoubleNormal : nullptr );
        if ( !parseRes.has_value() && ctx.cancel_group_execution() )
            parseError = std::move( parseRes.error() );

        pc.points[VertId( i )] = Vector3f( tempDoubleCoord - firstLineCoord );
        if ( hasNormals )
            pc.normals[VertId( i )] = Vector3f( tempDoubleNormal );
        if ( hasColors )
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
        .callback = settings.callback,
        .telemetrySignal = settings.telemetrySignal
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
    if ( hasBom( str ) )
        str = str.substr( 3 );

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

void telemetryLogSize( const PointCloud& cloud )
{
    TelemetrySignal( "Open Pnts Log Pnts " + std::to_string( intLog2( cloud.calcNumValidPoints() ) ) );
}

static void telemetryOpenPoints( const std::string& ext, const PointCloud& cloud, const PointsLoadSettings& settings )
{
    if ( !settings.telemetrySignal )
        return;

    std::string signalString = "Open " + ext;

    if ( cloud.validPoints.any() )
    {
        signalString += " VP";
        if ( cloud.hasNormals() )
            signalString += 'N';
        if ( settings.colors && settings.colors->size() >= cloud.points.size() )
            signalString += 'C';
    }

    if ( settings.outXf && *settings.outXf != AffineXf3f{} )
        signalString += " XF";

    TelemetrySignal( signalString );
    telemetryLogSize( cloud );
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

    auto res = loader.fileLoad( file, settings );
    if ( res )
        telemetryOpenPoints( ext, *res, settings );
    return res;
}

Expected<PointCloud> fromAnySupportedFormat( std::istream& in, const std::string& extension, const PointsLoadSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = getPointsLoader( ext );
    if ( !loader.streamLoad )
        return unexpectedUnsupportedFileExtension();

    auto res = loader.streamLoad( in, settings );
    if ( res )
        telemetryOpenPoints( ext, *res, settings );
    return res;
}

namespace
{

// file name prefixes used by the multi-scan laser export (e.g. _intemp000007.pose and _laser000007.ply)
constexpr std::string_view cScanPosePrefix = "_intemp";
constexpr std::string_view cScanPlyPrefix = "_laser";

// parses the integer index from a scan file stem with the given prefix, e.g. ("_intemp000007", "_intemp") -> 7;
// returns std::nullopt if the stem does not start with the prefix or the remainder is not a pure number
std::optional<int> parseScanIndex( const std::string& stem, std::string_view prefix )
{
    if ( !std::string_view( stem ).starts_with( prefix ) )
        return {};
    const char* first = stem.data() + prefix.size();
    const char* last = stem.data() + stem.size();
    int index = 0;
    const auto [ptr, ec] = std::from_chars( first, last, index );
    if ( ec != std::errc{} || ptr != last )
        return {};
    return index;
}

// reads a 4x4 row-major rigid transformation from a .pose file as AffineXf3f (the last row 0 0 0 1 is ignored)
Expected<AffineXf3f> readScanPose( const std::filesystem::path& file )
{
    std::ifstream in( file );
    if ( !in )
        return unexpected( "Cannot open file for reading " + utf8string( file ) );

    AffineXf3f xf;
    in  >> xf.A.x.x >> xf.A.x.y >> xf.A.x.z >> xf.b.x
        >> xf.A.y.x >> xf.A.y.y >> xf.A.y.z >> xf.b.y
        >> xf.A.z.x >> xf.A.z.y >> xf.A.z.z >> xf.b.z;
    if ( in.fail() )
        return unexpected( "Cannot parse transformation from " + utf8string( file ) );
    return xf;
}

} // anonymous namespace

Expected<PointCloud> fromMultiScanFolder( const std::filesystem::path& folder, const ProgressCallback& callback )
{
    MR_TIMER;

    std::error_code ec;
    if ( !std::filesystem::is_directory( folder, ec ) )
        return unexpected( utf8string( folder ) + " is not a folder" );

    // collect .pose and .ply files indexed by the number embedded in their names
    std::map<int, std::filesystem::path> poseFiles, plyFiles;
    for ( auto entry : Directory{ folder, ec } )
    {
        if ( !entry.is_regular_file( ec ) )
            continue;
        const auto path = entry.path();
        const auto stem = utf8string( path.stem() );
        auto ext = utf8string( path.extension() );
        for ( auto& c : ext )
            c = ( char )tolower( c );

        if ( ext == ".pose" )
        {
            if ( auto idx = parseScanIndex( stem, cScanPosePrefix ) )
                poseFiles[*idx] = path;
        }
        else if ( ext == ".ply" )
        {
            if ( auto idx = parseScanIndex( stem, cScanPlyPrefix ) )
                plyFiles[*idx] = path;
        }
    }

    // keep only the indices having both a .pose and a .ply file
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> pairs; // ( .pose, .ply )
    for ( const auto& [idx, posePath] : poseFiles )
    {
        if ( auto it = plyFiles.find( idx ); it != plyFiles.end() )
            pairs.emplace_back( posePath, it->second );
    }
    if ( pairs.empty() )
        return unexpected( "No pairs of " + std::string( cScanPosePrefix ) + "*.pose and "
            + std::string( cScanPlyPrefix ) + "*.ply files found in " + utf8string( folder ) );

    PointCloud res;
    for ( size_t i = 0; i < pairs.size(); ++i )
    {
        const auto& [posePath, plyPath] = pairs[i];

        auto xf = readScanPose( posePath );
        if ( !xf )
            return unexpected( std::move( xf.error() ) );

        auto cloud = fromPly( plyPath );
        if ( !cloud )
            return unexpected( std::move( cloud.error() ) );

        // transform the loaded points (and normals) into the common coordinate frame
        BitSetParallelFor( cloud->validPoints, [&] ( VertId v )
        {
            cloud->points[v] = ( *xf )( cloud->points[v] );
            if ( v < cloud->normals.size() )
                cloud->normals[v] = xf->A * cloud->normals[v];
        } );

        res.addPartByMask( *cloud, cloud->validPoints );

        if ( !reportProgress( callback, float( i + 1 ) / float( pairs.size() ) ) )
            return unexpectedOperationCanceled();
    }

    return res;
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
