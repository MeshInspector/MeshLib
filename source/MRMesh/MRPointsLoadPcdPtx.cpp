#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRAffineXf3.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRColor.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOParsing.h"
#include "MRString.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>

namespace MR::PointsLoad
{

namespace
{

/// calls the function for every token of the line separated by spaces or tabs, passing also the index of the token
template <typename F>
void forEachToken( std::string_view line, F&& f )
{
    constexpr std::string_view cSpaces = " \t\r\n";
    size_t index = 0;
    for ( auto pos = line.find_first_not_of( cSpaces ); pos != std::string_view::npos; ++index )
    {
        auto end = line.find_first_of( cSpaces, pos );
        if ( end == std::string_view::npos )
            end = line.size();
        f( index, line.substr( pos, end - pos ) );
        pos = line.find_first_not_of( cSpaces, end );
    }
}

std::vector<std::string_view> getTokens( std::string_view line )
{
    std::vector<std::string_view> res;
    forEachToken( line, [&]( size_t, std::string_view token ) { res.push_back( token ); } );
    return res;
}

template <typename T>
bool parseInteger( std::string_view token, T& value )
{
    const auto res = std::from_chars( token.data(), token.data() + token.size(), value );
    return res.ec == std::errc() && res.ptr == token.data() + token.size();
}

/// decompresses the data packed by LZF algorithm;
/// returns false if the input is corrupted or its decompressed size differs from outLen
bool lzfDecompress( const unsigned char* in, size_t inLen, unsigned char* out, size_t outLen )
{
    size_t ip = 0, op = 0;
    while ( ip < inLen )
    {
        const unsigned ctrl = in[ip++];
        if ( ctrl < 32 )
        {
            // literal run
            const size_t len = ctrl + 1;
            if ( ip + len > inLen || op + len > outLen )
                return false;
            std::memcpy( out + op, in + ip, len );
            ip += len;
            op += len;
            continue;
        }
        // copy of the bytes written before
        size_t len = ctrl >> 5;
        size_t back = ( size_t( ctrl & 0x1f ) << 8 ) + 1;
        if ( len == 7 )
        {
            if ( ip >= inLen )
                return false;
            len += in[ip++];
        }
        if ( ip >= inLen )
            return false;
        back += in[ip++];
        len += 2;
        if ( back > op || op + len > outLen )
            return false;
        for ( size_t i = 0; i < len; ++i, ++op )
            out[op] = out[op - back];
    }
    return op == outLen;
}

/// description of one field of a point record in PCD file
struct PcdField
{
    std::string name;
    int size = 4;         ///< bytes in one value
    char type = 'F';      ///< I - signed integer, U - unsigned integer, F - floating point
    size_t count = 1;     ///< values per point
    size_t offset = 0;    ///< bytes from the beginning of point record to this field
    size_t token = 0;     ///< index of the first value of this field in a line of text data
};

double readPcdValue( const char* p, const PcdField& f )
{
    auto read = [p]<typename T>( T v ) { std::memcpy( &v, p, sizeof( T ) ); return double( v ); };
    switch ( f.type )
    {
    case 'F':
        return f.size == 4 ? read( float() ) : read( double() );
    case 'I':
        return f.size == 1 ? read( std::int8_t() ) : f.size == 2 ? read( std::int16_t() ) : f.size == 4 ? read( std::int32_t() ) : read( std::int64_t() );
    default:
        return f.size == 1 ? read( std::uint8_t() ) : f.size == 2 ? read( std::uint16_t() ) : f.size == 4 ? read( std::uint32_t() ) : read( std::uint64_t() );
    }
}

/// rgb and rgba fields keep 0xAARRGGBB
Color pcdColor( std::uint32_t packed )
{
    return Color( int( ( packed >> 16 ) & 0xff ), int( ( packed >> 8 ) & 0xff ), int( packed & 0xff ) );
}

} // anonymous namespace

Expected<PointCloud> fromPcd( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPcd( in, settings ), file );
}

Expected<PointCloud> fromPcd( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;

    std::vector<PcdField> fields;
    size_t width = 0, height = 1;
    std::optional<size_t> pointsInHeader;
    std::string dataKind;

    std::string line;
    for ( bool first = true; dataKind.empty() && std::getline( in, line ); first = false )
    {
        std::string_view lineView = line;
        if ( first && hasBom( lineView ) )
            lineView = lineView.substr( 3 );
        const auto tokens = getTokens( lineView );
        if ( tokens.empty() || tokens[0].starts_with( '#' ) )
            continue;
        const auto key = tokens[0];
        const auto numValues = tokens.size() - 1;
        auto perField = [&]( auto&& set )
        {
            if ( numValues != fields.size() )
                return false;
            for ( size_t i = 0; i < numValues; ++i )
                if ( !set( fields[i], tokens[i + 1] ) )
                    return false;
            return true;
        };
        bool ok = true;
        if ( key == "FIELDS" || key == "COLUMNS" )
        {
            fields.resize( numValues );
            perField( []( PcdField& f, std::string_view v ) { f.name = v; return true; } );
        }
        else if ( key == "SIZE" )
            ok = perField( []( PcdField& f, std::string_view v ) { return parseInteger( v, f.size ) && ( f.size == 1 || f.size == 2 || f.size == 4 || f.size == 8 ); } );
        else if ( key == "TYPE" )
            ok = perField( []( PcdField& f, std::string_view v ) { f.type = v[0]; return v.size() == 1 && ( f.type == 'I' || f.type == 'U' || f.type == 'F' ); } );
        else if ( key == "COUNT" )
            ok = perField( []( PcdField& f, std::string_view v ) { return parseInteger( v, f.count ) && f.count > 0; } );
        else if ( key == "WIDTH" )
            ok = numValues == 1 && parseInteger( tokens[1], width );
        else if ( key == "HEIGHT" )
            ok = numValues == 1 && parseInteger( tokens[1], height );
        else if ( key == "POINTS" )
        {
            size_t n = 0;
            ok = numValues == 1 && parseInteger( tokens[1], n );
            pointsInHeader = n;
        }
        else if ( key == "DATA" )
        {
            ok = numValues == 1;
            if ( ok )
                dataKind = tokens[1];
        }
        else if ( key != "VERSION" && key != "VIEWPOINT" )
            ok = false;
        if ( !ok )
            return unexpected( "Invalid line in PCD header: " + line );
    }

    const bool ascii = dataKind == "ascii";
    const bool compressed = dataKind == "binary_compressed";
    if ( !ascii && !compressed && dataKind != "binary" )
        return unexpected( "Unsupported data kind in PCD file: " + dataKind );

    size_t pointSize = 0, numTokens = 0;
    for ( auto& f : fields )
    {
        if ( f.type == 'F' && f.size < 4 )
            return unexpected( "Unsupported type of field " + f.name + " in PCD file" );
        f.offset = pointSize;
        f.token = numTokens;
        pointSize += f.size * f.count;
        numTokens += f.count;
    }
    auto findField = [&]( std::string_view name ) -> const PcdField*
    {
        for ( const auto& f : fields )
            if ( f.name == name )
                return &f;
        return nullptr;
    };
    const PcdField* xyz[3] = { findField( "x" ), findField( "y" ), findField( "z" ) };
    if ( !xyz[0] || !xyz[1] || !xyz[2] )
        return unexpected( "No x, y, z fields in PCD file" );
    const PcdField* normal[3] = { findField( "normal_x" ), findField( "normal_y" ), findField( "normal_z" ) };
    const bool hasNormals = normal[0] && normal[1] && normal[2];
    const PcdField* rgb = findField( "rgb" );
    if ( !rgb )
        rgb = findField( "rgba" );
    const bool hasColors = settings.colors && rgb && rgb->size == 4;

    size_t numPoints = pointsInHeader ? *pointsInHeader : width * height;
    if ( numPoints == 0 )
        return unexpected( "No points in PCD file" );

    // binary data: records of points, or arrays of fields if compressed
    std::vector<char> data;
    // text data
    Buffer<char> text;
    std::vector<size_t> lineOffsets;

    const auto remainingSize = size_t( getStreamSize( in ) );
    if ( ascii )
    {
        auto maybeText = readCharBuffer( in );
        if ( !maybeText )
            return unexpected( std::move( maybeText.error() ) );
        text = std::move( *maybeText );
        lineOffsets = splitByLines( text.data(), text.size() );
        numPoints = std::min( numPoints, lineOffsets.size() - 1 );
    }
    else
    {
        if ( numPoints > remainingSize ) // every point takes at least one byte even after compression
            return unexpected( "PCD file is truncated" );
        const auto dataSize = numPoints * pointSize;
        if ( compressed )
        {
            std::uint32_t sizes[2] = {}; // compressed and uncompressed
            if ( !in.read( ( char* )sizes, sizeof( sizes ) ) || sizes[0] > remainingSize || sizes[1] != dataSize )
                return unexpected( "Invalid size of compressed data in PCD file" );
            std::vector<char> packed( sizes[0] );
            data.resize( dataSize );
            if ( !in.read( packed.data(), packed.size() )
                || !lzfDecompress( ( const unsigned char* )packed.data(), packed.size(), ( unsigned char* )data.data(), data.size() ) )
                return unexpected( "Cannot decompress data in PCD file" );
        }
        else
        {
            if ( dataSize > remainingSize )
                return unexpected( "PCD file is truncated" );
            data.resize( dataSize );
            if ( !in.read( data.data(), data.size() ) )
                return unexpected( "Cannot read data from PCD file" );
        }
    }

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpectedOperationCanceled();

    struct Record
    {
        Vector3d point;
        Vector3f normal;
        std::uint32_t rgb = 0;
    };
    /// returns false if the point is not valid
    auto readRecord = [&]( size_t i, Record& r ) -> bool
    {
        if ( ascii )
        {
            // 0..2 - coordinates, 3..5 - normal, 6 - color
            unsigned found = 0;
            const std::string_view textLine( text.data() + lineOffsets[i], lineOffsets[i + 1] - lineOffsets[i] );
            forEachToken( textLine, [&]( size_t index, std::string_view token )
            {
                float v = 0;
                for ( int k = 0; k < 3; ++k )
                {
                    if ( index == xyz[k]->token && parseSingleNumber<float>( token, v ) )
                    {
                        r.point[k] = v;
                        found |= 1u << k;
                    }
                    if ( hasNormals && index == normal[k]->token && parseSingleNumber<float>( token, v ) )
                    {
                        r.normal[k] = v;
                        found |= 8u << k;
                    }
                }
                if ( hasColors && index == rgb->token )
                {
                    if ( rgb->type == 'F' )
                    {
                        // the color is packed in the bits of a float number
                        if ( parseSingleNumber<float>( token, v ) )
                        {
                            std::memcpy( &r.rgb, &v, sizeof( v ) );
                            found |= 64;
                        }
                    }
                    else if ( std::int64_t n = 0; parseInteger( token, n ) )
                    {
                        r.rgb = std::uint32_t( n );
                        found |= 64;
                    }
                }
            } );
            if ( ( found & 7 ) != 7 )
                return false;
            if ( hasNormals && ( found & 56 ) != 56 )
                r.normal = {};
        }
        else
        {
            auto ptr = [&]( const PcdField& f )
            {
                return data.data() + ( compressed ? f.offset * numPoints + i * f.size * f.count : i * pointSize + f.offset );
            };
            for ( int k = 0; k < 3; ++k )
            {
                r.point[k] = readPcdValue( ptr( *xyz[k] ), *xyz[k] );
                if ( hasNormals )
                    r.normal[k] = float( readPcdValue( ptr( *normal[k] ), *normal[k] ) );
            }
            if ( hasColors )
                std::memcpy( &r.rgb, ptr( *rgb ), sizeof( r.rgb ) );
        }
        return std::isfinite( r.point.x ) && std::isfinite( r.point.y ) && std::isfinite( r.point.z );
    };

    // the first valid point becomes the origin, if the transformation is requested
    Vector3d shift;
    if ( settings.outXf )
    {
        Record r;
        for ( size_t i = 0; i < numPoints; ++i )
        {
            if ( readRecord( i, r ) )
            {
                shift = r.point;
                break;
            }
        }
        *settings.outXf = AffineXf3f::translation( Vector3f( shift ) );
    }

    PointCloud cloud;
    cloud.points.resizeNoInit( numPoints );
    cloud.validPoints.resize( numPoints, false );
    if ( hasNormals )
        cloud.normals.resizeNoInit( numPoints );
    if ( hasColors )
        settings.colors->resizeNoInit( numPoints );

    const auto keepGoing = BitSetParallelForAll( cloud.validPoints, [&]( VertId v )
    {
        Record r;
        const bool valid = readRecord( size_t( v ), r );
        cloud.points[v] = valid ? Vector3f( r.point - shift ) : Vector3f();
        if ( hasNormals )
            cloud.normals[v] = r.normal;
        if ( hasColors )
            ( *settings.colors )[v] = pcdColor( r.rgb );
        if ( valid )
            cloud.validPoints.set( v );
    }, subprogress( settings.callback, 0.5f, 1.0f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    if ( cloud.validPoints.none() )
        return unexpected( "Could not read any points" );
    return cloud;
}

Expected<PointCloud> fromPtx( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPtx( in, settings ), file );
}

Expected<PointCloud> fromPtx( std::istream& in, const PointsLoadSettings& settings )
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

    if ( !reportProgress( settings.callback, 0.4f ) )
        return unexpectedOperationCanceled();

    const auto lineOffsets = splitByLines( bufData, bufSize );
    const auto lineCount = lineOffsets.size() - 1;
    auto getLine = [&]( size_t i )
    {
        return std::string_view( bufData + lineOffsets[i], lineOffsets[i + 1] - lineOffsets[i] );
    };

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpectedOperationCanceled();

    // the file consists of one or several scans, each given in its own coordinates with the transformation to the common ones
    struct Scan
    {
        size_t firstLine = 0;  ///< the line of the first point
        size_t numPoints = 0;
        size_t firstPoint = 0; ///< the index of the first point in the resulting cloud
        Vector3d rows[4];      ///< 4x3 part of the matrix: common = x * rows[0] + y * rows[1] + z * rows[2] + rows[3]
    };
    std::vector<Scan> scans;
    size_t totalPoints = 0;
    for ( size_t ln = 0; ln < lineCount; )
    {
        if ( trim( getLine( ln ) ).empty() )
        {
            ++ln;
            continue;
        }
        // 2 lines of sizes, 4 lines of scanner position and axes, 4 lines of matrix
        constexpr size_t cHeaderLines = 10;
        int cols = 0, rows = 0;
        if ( ln + cHeaderLines > lineCount || !parseFirstNum( getLine( ln ), cols ) || !parseFirstNum( getLine( ln + 1 ), rows ) || cols < 0 || rows < 0 )
            return unexpected( "Invalid header of scan in PTX file" );
        Scan scan;
        for ( int i = 0; i < 4; ++i )
        {
            Color unused;
            if ( !parsePtsCoordinate( getLine( ln + 6 + i ), scan.rows[i], &unused ) )
                return unexpected( "Invalid matrix of scan in PTX file" );
        }
        scan.firstLine = ln + cHeaderLines;
        scan.numPoints = std::min( size_t( cols ) * size_t( rows ), lineCount - scan.firstLine ); // if the file is truncated
        scan.firstPoint = totalPoints;
        totalPoints += scan.numPoints;
        ln = scan.firstLine + scan.numPoints;
        scans.push_back( scan );
    }
    if ( totalPoints == 0 )
        return unexpected( "No points in PTX file" );

    // the position of the first scanner becomes the origin, if the transformation is requested
    Vector3d shift;
    if ( settings.outXf )
    {
        shift = scans[0].rows[3];
        *settings.outXf = AffineXf3f::translation( Vector3f( shift ) );
    }

    // colors are present if a point line has 7 numbers: x y z intensity r g b
    bool hasColors = false;
    if ( settings.colors )
    {
        for ( const auto& scan : scans )
        {
            if ( scan.numPoints == 0 )
                continue;
            hasColors = getTokens( getLine( scan.firstLine ) ).size() >= 7;
            break;
        }
    }

    PointCloud cloud;
    cloud.points.resizeNoInit( totalPoints );
    cloud.validPoints.resize( totalPoints, false );
    if ( hasColors )
        settings.colors->resizeNoInit( totalPoints );

    const auto keepGoing = BitSetParallelForAll( cloud.validPoints, [&]( VertId v )
    {
        // the last scan starting not after this point
        size_t s = scans.size() - 1;
        while ( scans[s].firstPoint > size_t( v ) )
            --s;
        const auto& scan = scans[s];

        Vector3d p;
        Color color;
        const bool parsed = parsePtsCoordinate( getLine( scan.firstLine + size_t( v ) - scan.firstPoint ), p, hasColors ? &color : nullptr ).has_value();
        // the directions without reflection are stored as zero points
        const bool valid = parsed && p != Vector3d();
        if ( valid )
        {
            p = p.x * scan.rows[0] + p.y * scan.rows[1] + p.z * scan.rows[2] + scan.rows[3] - shift;
            cloud.validPoints.set( v );
        }
        cloud.points[v] = valid ? Vector3f( p ) : Vector3f();
        if ( hasColors )
            ( *settings.colors )[v] = color;
    }, subprogress( settings.callback, 0.5f, 1.0f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    if ( cloud.validPoints.none() )
        return unexpected( "Could not read any points" );
    return cloud;
}

MR_ADD_POINTS_LOADER( IOFilter( "Point Cloud Data (.pcd)", "*.pcd" ), fromPcd )
MR_ADD_POINTS_LOADER( IOFilter( "Leica PTX (.ptx)", "*.ptx" ), fromPtx )

} // namespace MR::PointsLoad
