#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVoxelsSave.h"
#include "MRImageSave.h"
#include "MRFloatGrid.h"
#include "MRObjectVoxels.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include <fstream>
#include <filesystem>

namespace MR
{

namespace VoxelsSave
{
const IOFilters Filters = 
{
    {"Raw (.raw)","*.raw"}
};

tl::expected<void, std::string> saveRAW( const std::filesystem::path& path, const ObjectVoxels& voxelsObject, ProgressCallback callback )
{
    if ( path.empty() )
    {
        return tl::make_unexpected( "Path is empty" );
    }

    auto ext = utf8string( path.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    const auto& dims = voxelsObject.dimensions();
    if ( dims.x == 0 || dims.y == 0 || dims.z == 0 )
    {
        return tl::make_unexpected(  "ObjectVoxels is empty" );
    }

    auto parentPath = path.parent_path();
    std::error_code ec;
    if ( !std::filesystem::is_directory( parentPath, ec ) )
    {
        ec.clear();
        if ( !std::filesystem::create_directories( parentPath, ec ) )
        {
            std::stringstream ss;
            ss << "Cannot create directories: " << utf8string( parentPath ) << std::endl;
            ss << "Error: " << ec.value() << " Message: " << ec.message() << std::endl;
            return tl::make_unexpected( ss.str() );
        }
    }

    std::stringstream prefix;
    prefix.precision( 3 );
    prefix << "W" << dims.x << "_H" << dims.y << "_S" << dims.z;    // dims
    const auto& voxSize = voxelsObject.voxelSize();
    prefix << "_V" << voxSize.x * 1000.0f << "_" << voxSize.y * 1000.0f << "_" << voxSize.z * 1000.0f << "_F "; // voxel size "_F" for float
    prefix << utf8string( path.filename() );                        // name

    std::filesystem::path outPath = parentPath / prefix.str();
    std::ofstream outFile( outPath, std::ios::binary );
    if ( !outFile )
    {
        std::stringstream ss;
        ss << "Cannot write file: " << utf8string( outPath ) << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    const auto& grid = voxelsObject.grid();
    auto accessor = grid->getConstAccessor();

    // this coping block allow us to write data to disk faster
    std::vector<float> buffer( size_t( dims.x )*dims.y*dims.z );
    size_t dimsXY = size_t( dims.x )*dims.y;

    for ( int z = 0; z < dims.z; ++z )
    {
        for ( int y = 0; y < dims.y; ++y )
        {
            for ( int x = 0; x < dims.x; ++x )
                {
                buffer[z*dimsXY + y * dims.x + x] = accessor.getValue( {x,y,z} );
            }
        }
    }

    if ( !writeByBlocks( outFile, (const char*) buffer.data(), buffer.size() * sizeof( float ), callback ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );
    if ( !outFile )
    {
        std::stringstream ss;
        ss << "Cannot write file: " << utf8string( outPath ) << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> saveSliceToImage( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                             SlicePlain slicePlain, int sliceNumber, float min, float max, ProgressCallback callback/* = {} */)
{
    const auto& bounds = voxelsObject.getActiveBounds();
    const auto dims = bounds.size();
    const int textureWidth = dims[( slicePlain + 1 ) % 3];
    const int textureHeight = dims[( slicePlain + 2 ) % 3];

    std::vector<Color> texture( textureWidth * textureHeight );
    Vector3i activeVoxel;
    switch ( slicePlain )
    {
    case SlicePlain::XY:
        if ( sliceNumber > bounds.max.z )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { bounds.min.x, bounds.min.y, sliceNumber };
        break;
    case SlicePlain::YZ:
        if ( sliceNumber > bounds.max.x )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { sliceNumber, bounds.min.y, bounds.min.z };
        break;
    case SlicePlain::ZX:
        if ( sliceNumber > bounds.max.y )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { bounds.min.x, sliceNumber, bounds.min.z };
        break;
    default:
        return  tl::make_unexpected( "Slice plain is invalid" );
    }
 
    const auto& grid = voxelsObject.grid();
    const auto accessor = grid->getConstAccessor();
  
    for ( int i = 0; i < int( texture.size() ); ++i )
    {
        openvdb::Coord coord;
        coord[slicePlain] = sliceNumber;
        coord[( slicePlain + 1 ) % 3] = ( i % textureWidth ) + bounds.min[( slicePlain + 1 ) % 3];
        coord[( slicePlain + 2 ) % 3] = ( i / textureWidth ) + bounds.min[( slicePlain + 2 ) % 3];

        const auto val = accessor.getValue( coord );
        const float normedValue = ( val - min ) / ( max - min );
        texture[i] = Color( Vector3f::diagonal( normedValue ) );

        if ( ( i % 100 ) && callback && !callback( float( i ) / texture.size() ) )
            return tl::make_unexpected("Operation was canceled");
    }

    MeshTexture meshTexture ( { std::move( texture ), {textureWidth, textureHeight} } );
    if ( !ImageSave::toAnySupportedFormat( meshTexture, path ) )
        return  tl::make_unexpected( "Unable to save image" );
    
    if ( callback )
        callback( 1.0f );

    return {};
}

tl::expected<void, std::string> saveAllSlicesToImage( const std::filesystem::path& path, const ObjectVoxels& voxelsObject,
                                                             SlicePlain slicePlain, float min, float max, ProgressCallback callback/* = {}*/)
{
    const auto& bounds = voxelsObject.getActiveBounds();
    switch ( slicePlain )
    {
    case SlicePlain::XY:
        for ( int z = bounds.min.z; z < bounds.max.z; ++z )
        {
            const auto res = saveSliceToImage( path.string() + "/slice_" + std::to_string( z ) + ".png", voxelsObject, slicePlain, z, min, max );
            if ( !res )
                return res;

            if ( ( z % 100 ) && callback && !callback( float( z ) / bounds.size().z ) )
                return {};                
        }
        break;
    case SlicePlain::YZ:
        for ( int x = bounds.min.x; x < bounds.max.x; ++x )
        {
            const auto res = saveSliceToImage( path.string() + "/slice_" + std::to_string( x ) + ".png", voxelsObject, slicePlain, x, min, max );
            if ( !res )
                return res;

            if ( ( x % 100 ) && callback && !callback( float( x ) / bounds.size().x ) )
                return {};
        }
        break;
    case SlicePlain::ZX:
        for ( int y = bounds.min.y; y < bounds.max.y; ++y )
        {
            const auto res = saveSliceToImage( path.string() + "/slice_" + std::to_string( y ) + ".png", voxelsObject, slicePlain, y, min, max );
            if ( !res )
                return res;

            if ( ( y % 100 ) && callback && callback( float( y ) / bounds.size().y ) )
                return {};
        }
        break;
    default:
        return  tl::make_unexpected( "Slice plain is invalid" );
    }

    if ( callback )
        callback( 1.f );
    return {};
}

} // namespace VoxelsSave

} // namespace MR
#endif
