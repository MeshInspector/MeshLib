#include "MRLas.h"
#ifndef MRIOEXTRAS_NO_LAS
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsLoadSettings.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRFmt.h"

#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( push )
#pragma warning( disable: 5267 ) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <lazperf/lazperf.hpp>
#include <lazperf/readers.hpp>
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( pop )
#endif

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

            if ( hasNormals )
            {
                Vector3d normal;
                std::memcpy( &normal.x, buf.data() + LasPointSize[pointFormat], 3 * sizeof( double ) );
                result.normals.push_back( Vector3f( normal ) );
            }
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
#endif
