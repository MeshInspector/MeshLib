#include "MRDistanceMapLoad.h"
#include "MRTimer.h"
#include "MRDistanceMap.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include <filesystem>

namespace MR
{

namespace DistanceMapLoad
{

const IOFilters Filters =
{
    {"Raw (.raw)","*.raw"}
};

tl::expected<DistanceMap, std::string> loadRaw( const std::filesystem::path& path, ProgressCallback progressCb )
{
    MR_TIMER;

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

    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return tl::make_unexpected( "File " + utf8string( path ) + " does not exist" );
    
    std::ifstream inFile( path, std::ios::binary );
    const std::string readError = "Cannot read file: " + utf8string( path );
    if ( !inFile )
        return tl::make_unexpected( readError );

    uint64_t resolution[2] = {};
    if ( !inFile.read( ( char* )resolution, sizeof( resolution ) ) )
        return tl::make_unexpected( readError );

    DistanceMap dmap( resolution[0], resolution[1] );
    const size_t size = size_t( resolution[0] ) * size_t( resolution[1] );
    std::vector<float> buffer( size );

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), progressCb ) )
        return tl::make_unexpected( std::string( "Loading canceled" ) );

    if ( !inFile )
        return tl::make_unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );
    
    
    return dmap;
}


} // namespace DistanceMapLoad

} // namespace MR
