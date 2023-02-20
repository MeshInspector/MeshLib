#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVoxelsSave.h"
#include "MRMeshFwd.h"
#include "MRImageSave.h"
#include "MRFloatGrid.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRColor.h"
#include "MRMeshTexture.h"
#include "MRPch/MROpenvdb.h"
#include <fstream>
#include <filesystem>
#include "openvdb/io/File.h"

#if FMT_VERSION < 80000
    const std::string & runtime( const std::string & str )
{
    return str;
}
#else
auto runtime( const std::string & str )
{
    return fmt::runtime( str );
}
#endif

namespace MR
{

namespace VoxelsSave
{
const IOFilters Filters = 
{
    {"Raw (.raw)","*.raw"},
    {"OpenVDB (.vdb)","*.vdb"}
};

tl::expected<void, std::string> saveRaw( const std::filesystem::path& path, const VdbVolume& vdbVolume, ProgressCallback callback )
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

    const auto& dims = vdbVolume.dims;
    if ( dims.x == 0 || dims.y == 0 || dims.z == 0 )
    {
        return tl::make_unexpected(  "VdbVolume is empty" );
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
    const auto& voxSize = vdbVolume.voxelSize;
    prefix << "_V" << voxSize.x * 1000.0f << "_" << voxSize.y * 1000.0f << "_" << voxSize.z * 1000.0f; // voxel size "_F" for float
    prefix << "_G" << ( vdbVolume.data->getGridClass() == openvdb::GRID_LEVEL_SET ? "1" : "0" ) << "_F ";
    prefix << utf8string( path.filename() );                        // name

    std::filesystem::path outPath = parentPath / prefix.str();
    std::ofstream outFile( outPath, std::ios::binary );
    if ( !outFile )
    {
        std::stringstream ss;
        ss << "Cannot write file: " << utf8string( outPath ) << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    const auto& grid = vdbVolume.data;
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
 
tl::expected<void, std::string> toVdb( const std::filesystem::path& path, const VdbVolume& vdbVolume, ProgressCallback /*callback*/ /*= {} */ )
{
    openvdb::io::File file( path.string() );
    openvdb::FloatGrid::Ptr gridPtr = std::make_shared<openvdb::FloatGrid>();
    gridPtr->setTree( vdbVolume.data->treePtr() );
    openvdb::math::Transform::Ptr transform = std::make_shared<openvdb::math::Transform>();
    transform->preScale( { vdbVolume.voxelSize.x, vdbVolume.voxelSize.y, vdbVolume.voxelSize.z } );
    gridPtr->setTransform( transform );
    file.write( { gridPtr } );
    file.close();
    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const std::filesystem::path& path, const VdbVolume& vdbVolume,
                                                      ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext == ".raw" )
        return saveRaw( path, vdbVolume, callback );
    else if ( ext == ".vdb" )
        return toVdb( path, vdbVolume, callback );
    else
        return tl::make_unexpected( std::string( "unsupported file extension" ) );
}



tl::expected<void, std::string> saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlain& slicePlain, int sliceNumber, ProgressCallback callback )
{
    const auto& dims = vdbVolume.dims;
    const int textureWidth = dims[( slicePlain + 1 ) % 3];
    const int textureHeight = dims[( slicePlain + 2 ) % 3];

    std::vector<Color> texture( textureWidth * textureHeight );
    Vector3i activeVoxel;
    switch ( slicePlain )
    {
    case SlicePlain::XY:
        if ( sliceNumber > dims.z )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { 0, 0, sliceNumber };
        break;
    case SlicePlain::YZ:
        if ( sliceNumber > dims.x )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { sliceNumber, 0, 0 };
        break;
    case SlicePlain::ZX:
        if ( sliceNumber > dims.y )
            return  tl::make_unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { 0, sliceNumber, 0 };
        break;
    default:
        return  tl::make_unexpected( "Slice plain is invalid" );
    }
 
    const auto& grid = vdbVolume.data;
    const auto accessor = grid->getConstAccessor();
  
    for ( int i = 0; i < int( texture.size() ); ++i )
    {
        openvdb::Coord coord;
        coord[slicePlain] = sliceNumber;
        coord[( slicePlain + 1 ) % 3] = ( i % textureWidth );
        coord[( slicePlain + 2 ) % 3] = ( i / textureWidth );

        const auto val = accessor.getValue( coord );
        const float normedValue = ( val - vdbVolume.min ) / ( vdbVolume.max - vdbVolume.min );
        texture[i] = Color( Vector3f::diagonal( normedValue ) );

        if ( !reportProgress( callback, [&]{ return float( i ) / texture.size(); }, i, 128 ) )
            return tl::make_unexpected( "Operation was canceled" );
    }

    MeshTexture meshTexture( { { std::move( texture ), {textureWidth, textureHeight} } } );
    auto saveRes = ImageSave::toAnySupportedFormat( meshTexture, path );
    if ( !saveRes.has_value() )
        return tl::make_unexpected( saveRes.error() );
    
    if ( callback )
        callback( 1.0f );

    return {};
}


tl::expected<void, std::string> saveAllSlicesToImage( const SavingSettings& settings )
{
    switch ( settings.slicePlain )
    {
    case SlicePlain::XY:
    {
        const size_t maxNumChars = std::to_string( settings.vdbVolume.dims.z ).size();
        for ( int z = 0; z < settings.vdbVolume.dims.z; ++z )
        {
            const auto res = saveSliceToImage( settings.path / fmt::format( runtime( settings.format ), z, maxNumChars ), settings.vdbVolume, settings.slicePlain, z );
            if ( !res )
                return res;

            if ( settings.cb && !settings.cb( float( z ) / settings.vdbVolume.dims.z ) )
                return tl::make_unexpected( "Operation was canceled" );
        }
        break;
    }
    case SlicePlain::YZ:
    {
        const size_t maxNumChars = std::to_string( settings.vdbVolume.dims.x ).size();
        for ( int x = 0; x < settings.vdbVolume.dims.x; ++x )
        {
            const auto res = saveSliceToImage( settings.path / fmt::format( runtime( settings.format ), x, maxNumChars ), settings.vdbVolume, settings.slicePlain, x );
            if ( !res )
                return res;

            if ( settings.cb && !settings.cb( float( x ) / settings.vdbVolume.dims.x ) )
                return tl::make_unexpected( "Operation was canceled" );
        }
        break;
    }
    case SlicePlain::ZX:
    {
        const size_t maxNumChars = std::to_string( settings.vdbVolume.dims.y ).size();
        for ( int y = 0; y < settings.vdbVolume.dims.y; ++y )
        {
            const auto res = saveSliceToImage( settings.path / fmt::format( runtime( settings.format ), y, maxNumChars ), settings.vdbVolume, settings.slicePlain, y );
            if ( !res )
                return res;

            if ( settings.cb && !settings.cb( float( y ) / settings.vdbVolume.dims.y ) )
                return tl::make_unexpected( "Operation was canceled" );
        }
        break;
    }
    default:
        return  tl::make_unexpected( "Slice plain is invalid" );
    }

    if ( settings.cb )
        settings.cb( 1.f );
    return {};
}

} // namespace VoxelsSave

} // namespace MR
#endif
