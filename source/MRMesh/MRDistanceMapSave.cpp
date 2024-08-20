#include "MRDistanceMapSave.h"
#include "MRObjectDistanceMap.h"
#include "MRDistanceMap.h"
#include "MRStringConvert.h"
#include "MRImageSave.h"
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

VoidOrErrStr toRAW( const std::filesystem::path& path, const DistanceMap& dmap )
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

VoidOrErrStr toMrDistanceMap( const std::filesystem::path& path, const DistanceMap& dmap, const DistanceMapToWorld& params )
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

VoidOrErrStr toAnySupportedFormat( const std::filesystem::path& path, const DistanceMap& dmap, const AffineXf3f * xf )
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

VoidOrErrStr saveDistanceMapToImage( const DistanceMap& dm, const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    threshold = std::clamp( threshold, 0.f, 1.f );
    auto size = dm.numPoints();
    std::vector<Color> pixels( size );
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    // find min-max
    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        if ( val )
        {
            if ( *val < min )
                min = *val;
            if ( val > max )
                max = *val;
        }
    }

    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        pixels[i] = val ?
            Color( Vector3f::diagonal( ( max - *val ) / ( max - min ) * ( 1 - threshold ) + threshold ) ) :
            Color::black();
    }

    return ImageSave::toAnySupportedFormat( { pixels, { int( dm.resX() ), int( dm.resY() ) } }, filename );
}

} // namespace MR
