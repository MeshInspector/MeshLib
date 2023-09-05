#include "MRPointsLoad.h"
#if !defined( MRMESH_NO_LAS )
#include "MRColor.h"
#include "MRPointCloud.h"

#include <lasreader_las.hpp>

namespace
{

using namespace MR;

#define COLOR( R, G, B ) Color( 0x##R, 0x##G, 0x##B )
constexpr std::array<Color, 19> lasDefaultPalette = {
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
};
#undef COLOR

Color getColor( uint8_t classification )
{
    if ( classification < lasDefaultPalette.size() )
        return lasDefaultPalette.at( classification );
    else
        return Color::black();
}

Expected<PointCloud, std::string> process( LASreader& reader, VertColors* colors, ProgressCallback callback )
{
    const auto& header = reader.header;

    PointCloud result;
    result.points.reserve( header.number_of_point_records );
    if ( colors )
        colors->reserve( header.number_of_point_records );

    while ( reader.read_point() )
    {
        if ( result.points.size() % 4096 == 0 )
            reportProgress( callback, (float)result.points.size() / (float)header.number_of_point_records );

        const auto& point = reader.point;
        result.points.emplace_back(
            point.get_x(),
            point.get_y(),
            point.get_z()
        );

        if ( colors )
        {
            if ( point.have_rgb )
            {
                // convert 16-bit colors to 8-bit
                colors->emplace_back(
                    point.get_R() >> 8,
                    point.get_G() >> 8,
                    point.get_B() >> 8
                );
            }
            else
            {
                colors->emplace_back( getColor( point.get_classification() ) );
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
    LASreaderLAS reader;
    if ( !reader.open( file.c_str() ) )
        return unexpected( "Failed to open LAS file" );
    if ( !reader.header.check() )
        return unexpected( "Incorrect LAS file" );

    auto result = process( reader, colors, std::move( callback ) );
    reader.close();
    return result;
}

Expected<PointCloud, std::string> fromLas( std::istream& in, VertColors* colors, ProgressCallback callback )
{
    LASreaderLAS reader;
    if ( !reader.open( in ) )
        return unexpected( "Failed to open LAS file" );
    if ( !reader.header.check() )
        return unexpected( "Incorrect LAS file" );

    auto result = process( reader, colors, std::move( callback ) );
    reader.close();
    return result;
}

} // namespace MR::PointsLoad

#endif // !defined( MRMESH_NO_LAS )