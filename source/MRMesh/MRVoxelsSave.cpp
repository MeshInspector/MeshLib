#include "MRVoxelsSave.h"
#include "MRFloatGrid.h"
#include "MRObjectVoxels.h"
#include "MRStringConvert.h"
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

tl::expected<void, std::string> saveRAW( const std::filesystem::path& path, const ObjectVoxels& voxelsObject )
{
    if ( path.empty() )
    {
        return tl::make_unexpected( "Path is empty" );
    }

    auto ext = path.extension().u8string();
    for ( auto & c : ext )
        c = (char) tolower( c );

    if ( ext != u8".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << asString( ext ) << "\"" << std::endl;
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
            ss << "Cannot create directories: " << parentPath.string() << std::endl;
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
        ss << "Cannot write file: " << outPath.string() << std::endl;
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

    if ( !outFile.write( (const char*) buffer.data(), buffer.size() * sizeof( float ) ) )
    {
        std::stringstream ss;
        ss << "Cannot write file: " << outPath.string() << std::endl;
        return tl::make_unexpected( ss.str() );
    }
    return {};
}

} // namespace VoxelsSave

} // namespace MR
