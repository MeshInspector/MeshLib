#include "MRPointsLoad.h"
#if !defined( MRMESH_NO_LAS )
#include "MRColor.h"
#include "MRPointCloud.h"

#include <lasreader.hpp>

namespace
{

static const std::map<int, MR::Color> lasPalette = {
    { 1, MR::Color( 0xAA, 0xAA, 0xAA ) },
    { 2, MR::Color( 0xAA, 0x55, 0x00 ) },
    { 5, MR::Color( 0x00, 0xAA, 0x00 ) },
    { 6, MR::Color( 0xFF, 0x55, 0x55 ) },
};

MR::Color getColor( uint8_t classification )
{
    const auto it = lasPalette.find( classification );
    if ( it != lasPalette.end() )
        return it->second;
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

    PointCloud result;
    //result.points.reserve( reader->npoints );
    //result.points.reserve( reader->p_count );
    while ( reader->read_point() )
    {
        result.points.emplace_back(
            reader->point.get_x(),
            reader->point.get_y(),
            reader->point.get_z()
        );
        if ( colors )
        {
            if ( reader->point.have_rgb )
            {
                colors->emplace_back(
                    reader->point.get_R(),
                    reader->point.get_G(),
                    reader->point.get_B()
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