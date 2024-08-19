#include "MRDistanceMapLoad.h"
#include "MRTimer.h"
#include "MRDistanceMap.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRTiffIO.h"
#include <filesystem>
#include <fstream>

namespace MR
{

namespace DistanceMapLoad
{

const IOFilters Filters =
{
    {"Raw (.raw)","*.raw"},
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
    {"GeoTIFF (.tif,.tiff)","*.tif;*.tiff"},
#endif
    {"MRDistanceMap (.mrdistancemap)","*.mrdistancemap"}
};

Expected<DistanceMap> fromRaw( const std::filesystem::path& path, ProgressCallback progressCb )
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
        return unexpected( std::string( "Loading canceled" ) );

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );
    
    return dmap;
}

Expected<DistanceMap> fromMrDistanceMap( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb )
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

    if ( !inFile.read( ( char* )&params, sizeof( DistanceMapToWorld ) ) )
        return unexpected( readError );

    uint64_t resolution[2] = {};
    if ( !inFile.read( ( char* )resolution, sizeof( resolution ) ) )
        return unexpected( readError );

    DistanceMap dmap( resolution[0], resolution[1] );
    const size_t size = size_t( resolution[0] ) * size_t( resolution[1] );
    std::vector<float> buffer( size );

    if ( !readByBlocks( inFile, ( char* )buffer.data(), buffer.size() * sizeof( float ), progressCb ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( !inFile )
        return unexpected( readError );

    for ( size_t i = 0; i < size; ++i )
        dmap.set( int( i ), buffer[i] );

    return dmap;
}
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
Expected<DistanceMap> fromTiff( const std::filesystem::path& path, DistanceMapToWorld& params, ProgressCallback progressCb /*= {} */ )
{
    MR_TIMER;

    auto paramsExp = readTiffParameters( path );
    if ( !paramsExp.has_value() )
        return unexpected( paramsExp.error() );

    if ( progressCb && !progressCb( 0.2f ) )
        return unexpected( std::string( "Loading canceled" ) );

    DistanceMap res( paramsExp->imageSize.x, paramsExp->imageSize.y );
    RawTiffOutput output;
    output.bytes = ( uint8_t* )res.data();
    output.size = ( paramsExp->imageSize.x * paramsExp->imageSize.y ) * sizeof( float );

    AffineXf3f outXf;
    output.p2wXf = &outXf;
    auto readRes = readRawTiff( path, output );
    if ( !readRes.has_value() )
        return unexpected( readRes.error() );

    auto transposedM = outXf.A.transposed();
    params.orgPoint = outXf.b;
    params.pixelXVec = transposedM.x;
    params.pixelYVec = transposedM.y;
    params.direction = transposedM.z;

    if ( progressCb && !progressCb( 0.8f ) )
        return unexpected( std::string( "Loading canceled" ) );

    return res;
}
#endif

Expected<DistanceMap> fromAnySupportedFormat( const std::filesystem::path& path, DistanceMapToWorld* params, ProgressCallback progressCb )
{
    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    ext.insert( std::begin( ext ), '*' );
    Expected<DistanceMap> res = unexpected( std::string( "unsupported file extension" ) );

    auto itF = std::find_if( Filters.begin(), Filters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( itF == Filters.end() )
        return res;

    if ( ext == "*.raw" )
        return fromRaw( path, progressCb );
    

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_TIFF )
    if ( ext == "*.tif" || ext == "*.tiff" )
    {
        if ( params )
            return fromTiff( path, *params, progressCb );
        else
        {
            DistanceMapToWorld defaultParams;
            return fromTiff( path, defaultParams, progressCb );
        }
    }
#endif

    if ( params )
        return fromMrDistanceMap( path, *params, progressCb );

    DistanceMapToWorld defaultParams;
    return fromMrDistanceMap( path, defaultParams, progressCb );
}

} // namespace DistanceMapLoad

} // namespace MR
