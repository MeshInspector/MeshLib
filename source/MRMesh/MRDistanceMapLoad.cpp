#include "MRDistanceMapLoad.h"

#include "MRDistanceMap.h"
#include "MRIOFormatsRegistry.h"
#include "MRProgressReadWrite.h"
#include "MRStringConvert.h"
#include "MRTiffIO.h"
#include "MRTimer.h"

#include <filesystem>
#include <fstream>

namespace MR
{

namespace DistanceMapLoad
{

Expected<DistanceMap> fromRaw( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
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

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), settings.progress ) )
        return unexpectedOperationCanceled();

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );
    
    return dmap;
}

Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
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

    auto params = settings.distanceMapToWorld;
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

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), settings.progress ) )
        return unexpectedOperationCanceled();

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );

    return dmap;
}

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
Expected<DistanceMap> fromTiff( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
{
    MR_TIMER;

    auto paramsExp = readTiffParameters( path );
    if ( !paramsExp.has_value() )
        return unexpected( paramsExp.error() );

    if ( !reportProgress( settings.progress, 0.2f ) )
        return unexpectedOperationCanceled();

    DistanceMap res( paramsExp->imageSize.x, paramsExp->imageSize.y );
    RawTiffOutput output;
    output.bytes = ( uint8_t* )res.data();
    output.size = ( paramsExp->imageSize.x * paramsExp->imageSize.y ) * sizeof( float );

    AffineXf3f outXf;
    if ( settings.distanceMapToWorld )
        output.p2wXf = &outXf;

    auto readRes = readRawTiff( path, output );
    if ( !readRes.has_value() )
        return unexpected( readRes.error() );

    if ( settings.distanceMapToWorld )
        *settings.distanceMapToWorld = outXf;

    if ( !reportProgress( settings.progress, 0.8f ) )
        return unexpectedOperationCanceled();

    return res;
}
#endif

Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, const DistanceMapLoadSettings& settings )
{
    auto ext = toLower( utf8string( path.extension() ) );
    ext.insert( std::begin( ext ), '*' );

    auto loader = getDistanceMapLoader( ext );
    if ( !loader )
        return unexpectedUnsupportedFileExtension();

    return loader( path, settings );
}

MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "MRDistanceMap (.mrdistancemap)", "*.mrdistancemap" ), fromMrDistanceMap )
MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "Raw (.raw)", "*.raw" ), fromRaw )
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
MR_ADD_DISTANCE_MAP_LOADER( IOFilter( "GeoTIFF (.tif,.tiff)", "*.tif;*.tiff" ), fromTiff )
#endif

} // namespace DistanceMapLoad

} // namespace MR
