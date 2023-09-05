#include "MRPointsLoad.h"
#if !defined( MRMESH_NO_LAS )
#include "MRColor.h"
#include "MRPointCloud.h"

#include <lasreader.hpp>

namespace
{

#define COLOR( R, G, B ) MR::Color( 0x##R, 0x##G, 0x##B )
constexpr std::array<MR::Color, 19> lasDefaultPalette = {
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

MR::Color getColor( uint8_t classification )
{
    if ( classification < lasDefaultPalette.size() )
        return lasDefaultPalette.at( classification );
    else
        return MR::Color::black();
}

}

namespace MR::PointsLoad
{

Expected<PointCloud, std::string> fromLas( const std::filesystem::path& file, VertColors* colors, ProgressCallback callback )
{
    LASreadOpener opener;
    opener.set_file_name( file.string().c_str() );
    assert( opener.active() );

    std::unique_ptr<LASreader> reader( opener.open() );
    if ( !reader )
        return unexpected( "Failed to parse LAS header" );
    auto& header = reader->header;

    PointCloud result;
    result.points.reserve( header.number_of_point_records );
    if ( colors )
        colors->reserve( header.number_of_point_records );

    while ( reader->read_point() )
    {
        if ( result.points.size() % 4096 == 0 )
            reportProgress( callback, (float)result.points.size() / (float)header.number_of_point_records );

        result.points.emplace_back(
            reader->point.get_x(),
            reader->point.get_y(),
            reader->point.get_z()
        );
        if ( colors )
        {
            if ( reader->point.have_rgb )
            {
                // convert 16-bit colors to 8-bit
                colors->emplace_back(
                    reader->point.get_R() >> 8,
                    reader->point.get_G() >> 8,
                    reader->point.get_B() >> 8
                );
            }
            else
            {
                colors->emplace_back( getColor( reader->point.get_classification() ) );
            }
        }
    }

    result.validPoints.resize( result.points.size(), true );

    reader->close();

    return result;
}

} // namespace MR::PointsLoad

#endif // !defined( MRMESH_NO_LAS )