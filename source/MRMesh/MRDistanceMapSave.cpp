#include "MRDistanceMapSave.h"

#include "MRDistanceMap.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectDistanceMap.h"
#include "MRStringConvert.h"

#include <filesystem>
#include <fstream>

namespace MR
{

namespace DistanceMapSave
{

Expected<void> toRAW( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& )
{
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

    if ( dmap.numPoints() == 0 )
        return unexpected( "ObjectDistanceMap is empty" );

    std::ofstream outFile( path, std::ios::binary );
    const std::string writeError = "Cannot write file: " + utf8string( path );
    if ( !outFile )
        return unexpected( writeError );    

    size_t resolution[2] = { dmap.resX(), dmap.resY() };
    if ( !outFile.write( ( char* )resolution, sizeof( resolution ) ) )
        return unexpected( writeError );

    // this coping block allow us to write data to disk faster
    const size_t numPoints = dmap.numPoints();
    std::vector<float> buffer( numPoints );
    for ( size_t i = 0; i < numPoints; ++i )
        buffer[i] = dmap.getValue( i );

    if ( !outFile.write( ( const char* )buffer.data(), buffer.size() * sizeof( float ) ) )
        return unexpected( writeError );

    return {};
}

Expected<void> toMrDistanceMap( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings )
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

    if ( dmap.numPoints() == 0 )
        return unexpected( "ObjectDistanceMap is empty" );

    std::ofstream outFile( path, std::ios::binary );
    const std::string writeError = "Cannot write file: " + utf8string( path );
    if ( !outFile )
        return unexpected( writeError );

    DistanceMapToWorld params;
    if ( settings.xf )
        params = *settings.xf;
    if ( !outFile.write( ( const char* )&params, sizeof( DistanceMapToWorld ) ) )
        return unexpected( writeError );

    size_t resolution[2] = { dmap.resX(), dmap.resY() };
    if ( !outFile.write( ( const char* )resolution, sizeof( resolution ) ) )
        return unexpected( writeError );

    // this coping block allow us to write data to disk faster
    const size_t numPoints = dmap.numPoints();
    std::vector<float> buffer( numPoints );
    for ( size_t i = 0; i < numPoints; ++i )
        buffer[i] = dmap.getValue( i );

    if ( !outFile.write( ( const char* )buffer.data(), buffer.size() * sizeof( float ) ) )
        return unexpected( writeError );

    return {};
}

Expected<void> toAnySupportedFormat( const DistanceMap& dmap, const std::filesystem::path& path, const DistanceMapSaveSettings& settings )
{
    auto ext = toLower( utf8string( path.extension() ) );
    ext.insert( std::begin( ext ), '*' );

    auto saver = getDistanceMapSaver( ext );
    if ( !saver )
        return unexpectedUnsupportedFileExtension();

    return saver( dmap, path, settings );
}

MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "MRDistanceMap (.mrdistancemap)", "*.mrdistancemap" ), toMrDistanceMap )
MR_ADD_DISTANCE_MAP_SAVER( IOFilter( "Raw (.raw)", "*.raw" ), toRAW )

} // namespace DistanceMapSave

} // namespace MR
