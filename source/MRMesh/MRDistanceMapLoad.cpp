#include "MRDistanceMapLoad.h"

#include "MRDistanceMap.h"
#include "MRIOFormatsRegistry.h"
#include "MRProgressReadWrite.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#include <filesystem>
#include <fstream>

namespace MR
{

namespace DistanceMapLoad
{

Expected<DistanceMap> fromRaw( const std::filesystem::path& path, DistanceMapToWorld*, ProgressCallback progressCb )
{
    MR_TIMER;

    if ( path.empty() )
        return unexpected( "Path is empty" );

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return unexpected( ss.str() );
    }

    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return unexpected( "File " + utf8string( path ) + " does not exist" );
    
    std::ifstream inFile( path, std::ios::binary );
    const std::string readError = "Cannot read file: " + utf8string( path );
    if ( !inFile )
        return unexpected( readError );

    uint64_t resolution[2] = {};
    if ( !inFile.read( ( char* )resolution, sizeof( resolution ) ) )
        return unexpected( readError );
    const size_t size = size_t( resolution[0] ) * size_t( resolution[1] );
    const size_t fileSize = std::filesystem::file_size( path, ec );

    if ( size != ( fileSize - 2 * sizeof( uint64_t ) ) / sizeof( float ) )
        return unexpected( "File does not hold a distance map" );

    DistanceMap dmap( resolution[0], resolution[1] );    
    std::vector<float> buffer( size );

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), progressCb ) )
        return unexpectedOperationCanceled();

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );
    
    return dmap;
}

Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb )
{
    if ( path.empty() )
        return unexpected( "Path is empty" );

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext != ".mrdistancemap" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".mrdistancemap\" current \"" << ext << "\"" << std::endl;
        return unexpected( ss.str() );
    }

    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return unexpected( "File " + utf8string( path ) + " does not exist" );

    std::ifstream inFile( path, std::ios::binary );
    const std::string readError = "Cannot read file: " + utf8string( path );
    if ( !inFile )
        return unexpected( readError );

    if ( !params )
    {
        static DistanceMapToWorld defaultParams;
        params = &defaultParams;
    }
    if ( !inFile.read( ( char* )params, sizeof( DistanceMapToWorld ) ) )
        return unexpected( readError );

    uint64_t resolution[2] = {};
    if ( !inFile.read( ( char* )resolution, sizeof( resolution ) ) )
        return unexpected( readError );

    DistanceMap dmap( resolution[0], resolution[1] );
    const size_t size = size_t( resolution[0] ) * size_t( resolution[1] );
    std::vector<float> buffer( size );

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), progressCb ) )
        return unexpectedOperationCanceled();

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );

    return dmap;
}

Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb )
{
    auto ext = toLower( utf8string( path.extension() ) );
    ext.insert( std::begin( ext ), '*' );

    auto loader = getDistanceMapLoader( ext );
    if ( !loader )
        return unexpectedUnsupportedFileExtension();

    return loader( path, params, progressCb );
}

MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "MRDistanceMap (.mrdistancemap)", "*.mrdistancemap" ), fromMrDistanceMap )
MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "Raw (.raw)", "*.raw" ), fromRaw )

} // namespace DistanceMapLoad

} // namespace MR
