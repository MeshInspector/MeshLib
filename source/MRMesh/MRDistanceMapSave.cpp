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

tl::expected<void, std::string> toRAW( const std::filesystem::path& path, const DistanceMap& dmap )
{
    if ( path.empty() )
        return tl::make_unexpected( "Path is empty" );

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    if ( dmap.numPoints() == 0 )
        return tl::make_unexpected( "ObjectDistanceMap is empty" );

    std::ofstream outFile( path, std::ios::binary );
    const std::string writeError = "Cannot write file: " + utf8string( path );
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

tl::expected<void, std::string> toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmap, const DistanceMapToWorld* params )
{
    if ( path.empty() )
        return tl::make_unexpected( "Path is empty" );

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext != ".mrdistancemap" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".mrdistancemap\" current \"" << ext << "\"" << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    if ( dmap.numPoints() == 0 )
        return tl::make_unexpected( "ObjectDistanceMap is empty" );

    std::ofstream outFile( path, std::ios::binary );
    const std::string writeError = "Cannot write file: " + utf8string( path );
    if ( !outFile )
        return tl::make_unexpected( writeError );
    
    if ( !outFile.write( ( const char* )params, sizeof( DistanceMapToWorld ) ) )
        return tl::make_unexpected( writeError );

    size_t resolution[2] = { dmap.resX(), dmap.resY() };
    if ( !outFile.write( ( const char* )resolution, sizeof( resolution ) ) )
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

tl::expected<void, std::string> toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmap, const DistanceMapToWorld* params)
{
    auto ext = "*" + utf8string(path.extension());
    for ( auto& c : ext )
        c = ( char )tolower( c );
    
    auto itF = std::find_if( Filters.begin(), Filters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extension == ext;
    } );
    if ( itF == Filters.end() )
        return tl::make_unexpected( std::string( "unsupported file extension" ) );

    if ( itF->extension == "*.raw" )
        return toRAW( path, dmap );

    return toMrDistanceMap( path, dmap, params );
}

} // namespace DistanceMapSave

} // namespace MR
