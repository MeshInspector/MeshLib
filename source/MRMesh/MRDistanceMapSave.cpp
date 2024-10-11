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
    {"Raw (.raw)","*.raw"},
    {"MRDistanceMap (.mrdistancemap)","*.mrdistancemap"}
};

Expected<void> toRAW( const std::filesystem::path& path, const DistanceMap& dmap )
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

Expected<void> toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmap, const DistanceMapToWorld& params )
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

Expected<void> toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmap, const AffineXf3f * xf )
{
    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );
    
    ext.insert( std::begin( ext ), '*' );
    auto itF = std::find_if( Filters.begin(), Filters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( itF == Filters.end() )
        return unexpected( std::string( "unsupported file extension" ) );

    if ( ext == "*.raw" )
        return toRAW( path, dmap );

    return toMrDistanceMap( path, dmap, xf ? *xf : AffineXf3f{} );
}

} // namespace DistanceMapSave

} // namespace MR
