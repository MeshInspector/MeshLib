#include "MRDistanceMapSave.h"
#include "MRObjectDistanceMap.h"
#include "MRDistanceMap.h"
#include "MRStringConvert.h"
#include <fstream>
#include <filesystem>

namespace MR
{

namespace DistanceMapSave
{

const IOFilters Filters =
{
    {"Raw (.raw)","*.raw"}
};

tl::expected<void, std::string> saveRAW( const std::filesystem::path& path, const DistanceMap& dmap )
{
    if ( path.empty() )
        return tl::make_unexpected( "Path is empty" );

    auto ext = path.extension().u8string();
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext != u8".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << asString( ext ) << "\"" << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    if ( dmap.numPoints() == 0 )
        return tl::make_unexpected( "ObjectDistanceMap is empty" );

    std::ofstream outFile( path, std::ios::binary );
    const std::string writeError = "Cannot write file: " + path.string();
    if ( !outFile )
        return tl::make_unexpected( writeError );

    size_t resolution[2] = { dmap.resX(), dmap.resY() };
    if ( !outFile.write( ( char* )resolution, sizeof( resolution ) ) )
        return tl::make_unexpected( writeError );

    // this coping block allow us to write data to disk faster
    const size_t numPoints = dmap.numPoints();
    std::vector<float> buffer( numPoints );
    for ( size_t i = 0; i < numPoints; ++i )
        buffer[i] = dmap.getValue( i );

    if ( !outFile.write( ( const char* )buffer.data(), buffer.size() * sizeof( float ) ) )
        return tl::make_unexpected( writeError );

    return {};
}

} // namespace DistanceMapSave

} // namespace MR
