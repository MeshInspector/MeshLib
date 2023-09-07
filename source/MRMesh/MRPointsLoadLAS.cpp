#include "MRPointsLoad.h"
#if !defined( MRMESH_NO_LAS )
#include "MRColor.h"
#include "MRPointCloud.h"

#include "MRPch/MRSpdlog.h"

#include <lazperf/lazperf.hpp>
#include <lazperf/readers.hpp>

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

Expected<PointCloud, std::string> process( lazperf::reader::basic_file& reader, VertColors* colors, ProgressCallback callback )
{
    const auto pointCount = reader.pointCount();

    const auto& header = reader.header();
    const auto pointFormat = header.pointFormat();

    PointCloud result;
    result.points.reserve( pointCount );
    if ( colors )
        colors->reserve( pointCount );

    constexpr size_t maxPointRecordLength = sizeof( LasPoint10 );
    std::array<char, maxPointRecordLength> buf { '\0' };
    if ( buf.size() < header.point_record_length )
        return unexpected( fmt::format( "Unsupported LAS format version: {}.{}", header.version.major, header.version.minor ) );

    for ( auto i = 0; i < pointCount; ++i )
    {
        if ( i % 4096 == 0 )
            reportProgress( callback, (float)i / (float)pointCount );

        reader.readPoint( buf.data() );
        const auto point = getPoint( buf.data(), pointFormat );
        result.points.emplace_back(
            point.x * reader.header().scale.x + reader.header().offset.x,
            point.y * reader.header().scale.y + reader.header().offset.y,
            point.z * reader.header().scale.z + reader.header().offset.z
        );

        if ( colors )
        {
            if ( hasColorChannels( pointFormat ) )
            {
                const auto colorChannels = *getColorChannels( buf.data(), pointFormat );
                colors->emplace_back(
                    colorChannels.red,
                    colorChannels.green,
                    colorChannels.blue
                );
            }
            else
            {
                colors->emplace_back( getColor( getClassification( buf.data(), pointFormat ) ) );
            }
        }
    }

    result.validPoints.resize( result.points.size(), true );

    return result;
}

}

namespace MR::PointsLoad
{

Expected<PointCloud, std::string> fromLas( const std::filesystem::path& file, VertColors* colors, ProgressCallback callback )
{
    try
    {
        lazperf::reader::named_file reader( file.string() );
        return process( reader, colors, std::move( callback ) );
    }
    catch ( const std::exception& exc )
    {
        return unexpected( fmt::format( "Failed to read file: {}", exc.what() ) );
    }
}

Expected<PointCloud, std::string> fromLas( std::istream& in, VertColors* colors, ProgressCallback callback )
{
    try
    {
        lazperf::reader::generic_file reader( in );
        return process( reader, colors, std::move( callback ) );
    }
    catch ( const std::exception& exc )
    {
        return unexpected( fmt::format( "Failed to read file: {}", exc.what() ) );
    }
}

} // namespace MR::PointsLoad

#endif // !defined( MRMESH_NO_LAS )