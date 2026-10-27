#include "MRLas.h"
#ifndef MRIOEXTRAS_NO_LAS
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRChrono.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsLoadSettings.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVector.h"
#include "MRPch/MRFmt.h"

#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( push )
#pragma warning( disable: 5267 ) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <lazperf/filestream.hpp>
#include <lazperf/lazperf.hpp>
#include <lazperf/readers.hpp>
#include <lazperf/vlr.hpp>
#include <lazperf/writers.hpp>
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( pop )
#endif

#include <algorithm>
#include <cmath>
#include <fstream>

namespace
{

/*
 * LAS file format structures (not exposed by LAZperf)
 * Specification: https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
 */

// point data structures
#pragma pack(push, 1)
struct LasPoint
{
    int32_t x;
    int32_t y;
    int32_t z;
};

// LAS 1.0

struct LasPoint0
{
    int32_t x;
    int32_t y;
    int32_t z;
    uint16_t intensity;
    uint8_t returnNumber : 3;
    uint8_t numberOfReturns : 3;
    uint8_t scanDirectionFlag : 1;
    uint8_t edgeOfFlightLine : 1;
    uint8_t classification;
    int8_t scanAngle;
    uint8_t userData;
    uint16_t pointSourceId;
};

struct GpsTime
{
    double gpsTime;
};

struct ColorChannels
{
    uint16_t red;
    uint16_t green;
    uint16_t blue;
};

struct WavePackets
{
    uint8_t wavePacketDescriptorIndex;
    uint64_t byteOffsetToWaveformData;
    uint32_t waveformPacketSize;
    float returnPointWaveformLocation;
    float parametricDx;
    float parametricDy;
    float parametricDz;
};

struct LasPoint1 : public LasPoint0, public GpsTime {};
struct LasPoint2 : public LasPoint0, public ColorChannels {};
struct LasPoint3 : public LasPoint0, public GpsTime, public ColorChannels {};
struct LasPoint4 : public LasPoint0, public GpsTime, public WavePackets {};
struct LasPoint5 : public LasPoint0, public GpsTime, public ColorChannels, public WavePackets {};

// LAS 1.4

struct LasPoint6
{
    int32_t x;
    int32_t y;
    int32_t z;
    uint16_t intensity;
    uint8_t returnNumber : 4;
    uint8_t numberOfReturns : 4;
    uint8_t classificationFlags : 4;
    uint8_t scannerChannel : 2;
    uint8_t scanDirectionFlag : 1;
    uint8_t edgeOfFlightLine : 1;
    uint8_t classification;
    uint8_t userData;
    int16_t scanAngle;
    uint16_t pointSourceId;
    double gpsTime;
};

struct NirChannel
{
    uint16_t nir;
};

struct LasPoint7 : public LasPoint6, public ColorChannels {};
struct LasPoint8 : public LasPoint6, public ColorChannels, public NirChannel {};
struct LasPoint9 : public LasPoint6, public WavePackets {};
struct LasPoint10 : public LasPoint6, public ColorChannels, public NirChannel, public WavePackets {};

struct ExtraBytes
{
    unsigned char reserved[2];
    unsigned char data_type;
    unsigned char options;
    char name[32];
    unsigned char unused[4];
    char no_data[8];
    unsigned char deprecated1[16];
    char min[8];
    unsigned char deprecated2[16];
    char max[8];
    unsigned char deprecated3[16];
    double scale;
    unsigned char deprecated4[16];
    double offset;
    unsigned char deprecated5[16];
    char description[32];
};

#pragma pack(pop)

static_assert( sizeof( LasPoint0 ) == 20, "Incorrect struct size" );
static_assert( sizeof( LasPoint1 ) == 28, "Incorrect struct size" );
static_assert( sizeof( LasPoint2 ) == 26, "Incorrect struct size" );
static_assert( sizeof( LasPoint3 ) == 34, "Incorrect struct size" );
static_assert( sizeof( LasPoint4 ) == 57, "Incorrect struct size" );
static_assert( sizeof( LasPoint5 ) == 63, "Incorrect struct size" );
static_assert( sizeof( LasPoint6 ) == 30, "Incorrect struct size" );
static_assert( sizeof( LasPoint7 ) == 36, "Incorrect struct size" );
static_assert( sizeof( LasPoint8 ) == 38, "Incorrect struct size" );
static_assert( sizeof( LasPoint9 ) == 59, "Incorrect struct size" );
static_assert( sizeof( LasPoint10 ) == 67, "Incorrect struct size" );
static_assert( sizeof( ExtraBytes ) == 192, "Incorrect struct size" );

constexpr size_t LasPointSize[] =
{
    sizeof( LasPoint0 ),
    sizeof( LasPoint1 ),
    sizeof( LasPoint2 ),
    sizeof( LasPoint3 ),
    sizeof( LasPoint4 ),
    sizeof( LasPoint5 ),
    sizeof( LasPoint6 ),
    sizeof( LasPoint7 ),
    sizeof( LasPoint8 ),
    sizeof( LasPoint9 ),
    sizeof( LasPoint10 )
};

LasPoint getPoint( const char* buf, int )
{
    // TODO: fix endianness
    return *reinterpret_cast<const LasPoint*>( buf );
}

uint8_t getClassification( const char* buf, int format )
{
    if ( 0 <= format && format <= 5 )
        return reinterpret_cast<const LasPoint0*>( buf )->classification;
    else if ( 6 <= format && format <= 10 )
        return reinterpret_cast<const LasPoint6*>( buf )->classification;
    else
        return 0;
}

bool hasColorChannels( int format )
{
    switch ( format )
    {
        case 0:
        case 1:
        case 4:
        case 6:
        case 9:
            return false;
        case 2:
        case 3:
        case 5:
        case 7:
        case 8:
        case 10:
            return true;
        default:
            return false;
    }
}

std::optional<ColorChannels> getColorChannels( const char* buf, int format )
{
    // TODO: fix endianness
    switch ( format )
    {
        case 0:
        case 1:
        case 4:
        case 6:
        case 9:
            return std::nullopt;
        case 2:
            return *reinterpret_cast<const LasPoint2*>( buf );
        case 3:
        case 5:
            return *reinterpret_cast<const LasPoint3*>( buf );
        case 7:
        case 8:
        case 10:
            return *reinterpret_cast<const LasPoint7*>( buf );
        default:
            return std::nullopt;
    }
}

using namespace MR;

// default LAS classification palette
constexpr std::array<Color, 19> lasDefaultPalette = {
#define COLOR( R, G, B ) Color( 0x##R, 0x##G, 0x##B )
    COLOR( BA, BA, BA ),
    COLOR( AA, AA, AA ),
    COLOR( AA, 55, 00 ),
    COLOR( 00, AA, AA ),
    COLOR( 55, FF, 55 ),
    COLOR( 00, AA, 00 ),
    COLOR( FF, 55, 55 ),
    COLOR( AA, 00, 00 ),
    COLOR( 55, 55, 55 ),
    COLOR( 55, FF, FF ),
    COLOR( AA, 00, AA ),
    COLOR( 00, 00, 00 ),
    COLOR( 55, 55, 55 ),
    COLOR( FF, FF, 55 ),
    COLOR( FF, FF, 55 ),
    COLOR( FF, 55, FF ),
    COLOR( FF, FF, 55 ),
    COLOR( 55, 55, FF ),
    COLOR( 64, 64, 64 ),
#undef COLOR
};

Color getColor( uint8_t classification )
{
    if ( classification < lasDefaultPalette.size() )
        return lasDefaultPalette.at( classification );
    else
        return Color::black();
}

Expected<PointCloud> process( lazperf::reader::basic_file& reader, const PointsLoadSettings& settings )
{
    const auto pointCount = reader.pointCount();

    const auto& header = reader.header();
    const auto pointFormat = header.pointFormat();
    if ( pointFormat < 0 || pointFormat > 10 )
        return unexpected( fmt::format( "Unsupported LAS point format: {}", pointFormat ) );

    if ( LasPointSize[pointFormat] > header.point_record_length )
        return unexpected( fmt::format( "Too short LAS point record length {} for point format {}, expected length {}",
            header.point_record_length, pointFormat, LasPointSize[pointFormat] ) );

    const auto extraBytesVlr = reader.vlrData( "LASF_Spec", 4 );
    bool hasNormals = false;
    if ( extraBytesVlr.size() == 3 * sizeof( ExtraBytes ) )
    {
        ExtraBytes extraBytes[3];
        for ( int i = 0; i < 3; ++i )
            std::memcpy( extraBytes + i, extraBytesVlr.data() + i * sizeof( ExtraBytes ), sizeof( ExtraBytes ) );
        if ( extraBytes[0].data_type == 10 && extraBytes[1].data_type == 10 && extraBytes[2].data_type == 10 ) // all extra types are doubles
        {
            // https://github.com/ASPRSorg/LAS/issues/37#issuecomment-1695757865
            // enough to check first field only
            hasNormals = strcmp( extraBytes[0].name, "NormalX" ) == 0 ||
                 strcmp( extraBytes[0].name, "nx" ) == 0 ||
                 strcmp( extraBytes[0].name, "normal_x" ) == 0 ||
                 strcmp( extraBytes[0].name, "normalx" ) == 0 ||
                 strcmp( extraBytes[0].name, "normal x" ) == 0;
        }
    }
    if ( hasNormals && LasPointSize[pointFormat] + 3 * sizeof( double ) > header.point_record_length )
        return unexpected( fmt::format( "Too short LAS point+normal record length {} for point format {}, expected length {}",
            header.point_record_length, pointFormat, LasPointSize[pointFormat] + 3 * sizeof( double ) ) );

    PointCloud result;
    result.points.reserve( pointCount );
    if ( settings.colors )
        settings.colors->reserve( pointCount );
    if ( hasNormals )
        result.normals.reserve( pointCount );

    Vector3d offset {
        header.offset.x,
        header.offset.y,
        header.offset.z,
    };
    if ( settings.outXf )
    {
        const Box3d box {
            { header.minx, header.miny, header.minz },
            { header.maxx, header.maxy, header.maxz },
        };
        const auto center = box.center();
        *settings.outXf = AffineXf3f::translation( Vector3f( center ) );
        offset -= center;
    }

    auto colorsHave16Bits = false;
    std::optional<VertColors> colorsHi;
    if ( settings.colors )
        colorsHi.emplace();

    std::vector<char> buf( header.point_record_length, '\0' );
    for ( auto i = 0; i < pointCount; ++i )
    {
        if ( i % 4096 == 0 )
            reportProgress( settings.callback, (float)i / (float)pointCount );

        reader.readPoint( buf.data() );
        const auto point = getPoint( buf.data(), pointFormat );
        const Vector3d pos {
            point.x * header.scale.x + offset.x,
            point.y * header.scale.y + offset.y,
            point.z * header.scale.z + offset.z,
        };
        result.points.emplace_back( pos );

        if ( settings.colors )
        {
            if ( hasColorChannels( pointFormat ) )
            {
                const auto colorChannels = *getColorChannels( buf.data(), pointFormat );
                // LAS stores color data in 16-bit per channel format, but most programs use 8-bit per channel.
                // Some of them convert color values to the 16-bit format (by multiplying by 256), some save them as is.
                // We have to support both approaches.
                const Color colorLo {
                    colorChannels.red % 0x100,
                    colorChannels.green % 0x100,
                    colorChannels.blue % 0x100,
                };
                const Color colorHi {
                    colorChannels.red >> 8,
                    colorChannels.green >> 8,
                    colorChannels.blue >> 8,
                };
                colorsHave16Bits |= ( colorHi.r || colorHi.g || colorHi.b );
                settings.colors->emplace_back( colorLo );
                colorsHi->emplace_back( colorHi );
            }
            else
            {
                const auto color = getColor( getClassification( buf.data(), pointFormat ) );
                settings.colors->emplace_back( color );
                colorsHi->emplace_back( color );
            }
        }

        if ( hasNormals )
        {
            Vector3d normal;
            std::memcpy( &normal.x, buf.data() + LasPointSize[pointFormat], 3 * sizeof( double ) );
            result.normals.push_back( Vector3f( normal ) );
        }
    }

    if ( settings.colors && colorsHave16Bits )
    {
        std::swap( *settings.colors, *colorsHi );
        colorsHi.reset();
    }

    result.validPoints.resize( result.points.size(), true );

    return result;
}

Expected<void> save( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings, bool compress )
{
    MR_TIMER;
    const size_t totalPoints = settings.onlyValidPoints ? cloud.validPoints.count() : cloud.points.size();
    if ( totalPoints > std::numeric_limits<uint32_t>::max() )
        return unexpected( fmt::format( "Too many points for LAS 1.2 format: {}", totalPoints ) );

    const bool saveColors = settings.colors && !settings.colors->empty();
    const bool saveNormals = cloud.points.size() <= cloud.normals.size();
    const int pointFormat = saveColors ? 2 : 0;
    const int ebCount = saveNormals ? int( 3 * sizeof( double ) ) : 0;

    Box3d box;
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
        if ( !settings.onlyValidPoints || cloud.validPoints.test( v ) )
            box.include( applyDouble( settings.xf, cloud.points[v] ) );
    if ( !box.valid() )
        box = Box3d( Vector3d(), Vector3d() );

    // the largest power of 10 scale keeping all coordinates relative to the center within int32 range
    const Vector3d offset = box.center();
    const double maxHalfSize = std::max( { box.size().x, box.size().y, box.size().z } ) / 2;
    const double scale = maxHalfSize > 0 ? std::pow( 10.0, std::ceil( std::log10( maxHalfSize / 2e9 ) ) ) : 1.0;
    const auto quantize = [&] ( const Vector3d& p )
    {
        return LasPoint {
            int32_t( std::lround( ( p.x - offset.x ) / scale ) ),
            int32_t( std::lround( ( p.y - offset.y ) / scale ) ),
            int32_t( std::lround( ( p.z - offset.z ) / scale ) ),
        };
    };
    const auto boxMin = quantize( box.min );
    const auto boxMax = quantize( box.max );

    lazperf::header12 header;
    constexpr char generatingSoftware[] = "MeshLib";
    std::memcpy( header.generating_software, generatingSoftware, sizeof( generatingSoftware ) );
    if ( const auto tm = Localtime( std::time( nullptr ) ) )
    {
        header.creation.day = uint16_t( tm->tm_yday + 1 );
        header.creation.year = uint16_t( tm->tm_year + 1900 );
    }
    header.header_size = uint16_t( lazperf::header12::Size );
    header.point_format_id = uint8_t( pointFormat );
    header.point_record_length = uint16_t( LasPointSize[pointFormat] + ebCount );
    header.point_count = uint32_t( totalPoints );
    header.points_by_return[0] = header.point_count;
    header.scale = { scale, scale, scale };
    header.offset = { offset.x, offset.y, offset.z };
    header.minx = boxMin.x * scale + offset.x;
    header.miny = boxMin.y * scale + offset.y;
    header.minz = boxMin.z * scale + offset.z;
    header.maxx = boxMax.x * scale + offset.x;
    header.maxy = boxMax.y * scale + offset.y;
    header.maxz = boxMax.z * scale + offset.z;

    const lazperf::laz_vlr lazVlr( pointFormat, ebCount, lazperf::DefaultChunkSize );
    lazperf::eb_vlr ebVlr;
    if ( saveNormals )
    {
        for ( const char* name : { "NormalX", "NormalY", "NormalZ" } )
        {
            lazperf::eb_vlr::ebfield field;
            field.data_type = 10; // double
            field.name = name;
            ebVlr.addField( field );
        }
    }

    header.point_offset = header.header_size;
    if ( compress )
    {
        header.point_format_id |= 0x80;
        header.point_offset += uint32_t( lazperf::vlr_header::Size + lazVlr.size() );
        ++header.vlr_count;
    }
    if ( saveNormals )
    {
        header.point_offset += uint32_t( lazperf::vlr_header::Size + ebVlr.size() );
        ++header.vlr_count;
    }

    header.write( out );
    if ( compress )
    {
        lazVlr.header().write( out );
        lazVlr.write( out );
    }
    if ( saveNormals )
    {
        ebVlr.header().write( out );
        ebVlr.write( out );
    }

    // compressed chunks are kept in memory, since the offset of the chunk table must precede them
    std::optional<lazperf::writer::chunk_compressor> compressor;
    std::vector<unsigned char> compressed;
    std::vector<lazperf::chunk> chunks;
    const auto finishChunk = [&]
    {
        const auto data = compressor->done();
        compressed.insert( compressed.end(), data.begin(), data.end() );
        chunks.back().offset = data.size();
        compressor.reset();
    };

    std::optional<Matrix3d> normXf;
    if ( saveNormals && settings.xf )
        normXf = settings.xf->A.inverse().transposed();

    std::vector<char> buf( header.point_record_length, '\0' );
    size_t numSaved = 0;
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.onlyValidPoints && !cloud.validPoints.test( v ) )
            continue;

        LasPoint2 point {};
        const auto pos = quantize( applyDouble( settings.xf, cloud.points[v] ) );
        point.x = pos.x;
        point.y = pos.y;
        point.z = pos.z;
        point.returnNumber = 1;
        point.numberOfReturns = 1;
        if ( saveColors )
        {
            const auto c = getAt( *settings.colors, v );
            point.red = uint16_t( c.r * 257 );
            point.green = uint16_t( c.g * 257 );
            point.blue = uint16_t( c.b * 257 );
        }
        std::memcpy( buf.data(), &point, LasPointSize[pointFormat] );

        if ( saveNormals )
        {
            Vector3d normal( cloud.normals[v] );
            if ( normXf )
                normal = *normXf * normal;
            std::memcpy( buf.data() + LasPointSize[pointFormat], &normal.x, 3 * sizeof( double ) );
        }

        if ( compress )
        {
            if ( !compressor )
            {
                compressor.emplace( pointFormat, ebCount );
                chunks.push_back( {} );
            }
            compressor->compress( buf.data() );
            if ( ++chunks.back().count == lazperf::DefaultChunkSize )
                finishChunk();
        }
        else
        {
            out.write( buf.data(), buf.size() );
        }

        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( compress )
    {
        if ( compressor )
            finishChunk();
        const auto chunkTableOffset = int64_t( header.point_offset + sizeof( int64_t ) + compressed.size() );
        out.write( (const char*)&chunkTableOffset, sizeof( chunkTableOffset ) );
        out.write( (const char*)compressed.data(), compressed.size() );

        const uint32_t chunkTableVersion = 0;
        const auto numChunks = uint32_t( chunks.size() );
        out.write( (const char*)&chunkTableVersion, sizeof( chunkTableVersion ) );
        out.write( (const char*)&numChunks, sizeof( numChunks ) );
        lazperf::OutFileStream stream( out );
        lazperf::compress_chunk_table( stream.cb(), chunks, false );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in LAS-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

}

namespace MR::PointsLoad
{

Expected<PointCloud> fromLas( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    try
    {
        lazperf::reader::named_file reader( utf8string( file ) );
        return process( reader, settings );
    }
    catch ( const std::exception& exc )
    {
        return unexpected( fmt::format( "Failed to read file: {}", exc.what() ) );
    }
}

Expected<PointCloud> fromLas( std::istream& in, const PointsLoadSettings& settings )
{
    try
    {
        lazperf::reader::generic_file reader( in );
        return process( reader, settings );
    }
    catch ( const std::exception& exc )
    {
        return unexpected( fmt::format( "Failed to read file: {}", exc.what() ) );
    }
}

MR_ADD_POINTS_LOADER( IOFilter( "LAS (.las)", "*.las" ), fromLas )
MR_ADD_POINTS_LOADER( IOFilter( "LASzip (.laz)", "*.laz" ), fromLas )

} // namespace MR::PointsLoad

namespace MR::PointsSave
{

Expected<void> toLas( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toLas( points, out, settings );
}

Expected<void> toLas( const PointCloud& points, std::ostream& out, const SaveSettings& settings )
{
    return save( points, out, settings, false );
}

Expected<void> toLaz( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toLaz( points, out, settings );
}

Expected<void> toLaz( const PointCloud& points, std::ostream& out, const SaveSettings& settings )
{
    return save( points, out, settings, true );
}

MR_ADD_POINTS_SAVER( IOFilter( "LAS (.las)", "*.las" ), toLas )
MR_ADD_POINTS_SAVER( IOFilter( "LASzip (.laz)", "*.laz" ), toLaz )

} // namespace MR::PointsSave
#endif
