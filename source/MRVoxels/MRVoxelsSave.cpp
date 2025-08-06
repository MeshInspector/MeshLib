#include "MRVoxelsSave.h"
#include "MROpenVDB.h"
#include "MRObjectVoxels.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"

#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRProgressReadWrite.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSerializer.h"

#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

#include <openvdb/io/Stream.h>

#include <fstream>
#include <filesystem>
#include <sstream>

namespace MR
{

namespace VoxelsSave
{

Expected<void> toRawFloat( const SimpleVolume& simpleVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER;
    if ( !writeByBlocks( out, (const char*) simpleVolume.data.data(), simpleVolume.data.size() * sizeof( float ), callback ) )
        return unexpectedOperationCanceled();
    if ( !out )
        return unexpected( std::string( "Stream write error" ) );
    return {};
}

Expected<void> toRawFloat( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER;
    return vdbVolumeToSimpleVolume( vdbVolume, {}, subprogress( callback, 0.0f, 0.2f ) ).and_then(
        [&out, sp = subprogress( callback, 0.2f, 1.0f )]( auto && sv )
        {
            return toRawFloat( sv, out, sp );
        }
    );
}

Expected<void> gridToRawFloat( const FloatGrid& grid, const Vector3i& dims, std::ostream& out, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;
    auto vdbVolume = floatGridToVdbVolume( grid );
    vdbVolume.dims = dims;
    return vdbVolumeToSimpleVolume( vdbVolume, {}, subprogress( callback, 0.0f, 0.2f ) ).and_then(
        [&out, sp = subprogress( callback, 0.2f, 1.0f )] ( auto&& sv )
    {
        return toRawFloat( sv, out, sp );
    }
    );
}

namespace
{

struct NamedOutFileStream
{
    std::filesystem::path file;
    std::ofstream out;
};

Expected<NamedOutFileStream> openRawAutonameStream( const Vector3i & dims, const Vector3f & voxSize, bool normalPlusGrad, const std::filesystem::path& file )
{
    if ( file.empty() )
        return unexpected( "Filename is empty" );

    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return unexpected( ss.str() );
    }

    if ( dims.x == 0 || dims.y == 0 || dims.z == 0 )
        return unexpected( "Volume is empty" );

    std::stringstream prefix;
    prefix.precision( 3 );
    prefix << "W" << dims.x << "_H" << dims.y << "_S" << dims.z;
    prefix << "_V" << voxSize.x * 1000.0f << "_" << voxSize.y * 1000.0f << "_" << voxSize.z * 1000.0f; // voxel size "_F" for float
    prefix << "_G" << ( normalPlusGrad ? "1" : "0" ) << "_F ";
    prefix << utf8string( file.filename() );

    NamedOutFileStream res;
    res.file = file;
    res.file.replace_filename( prefix.str() );
    res.out = std::ofstream( res.file, std::ios::binary );
    if ( !res.out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( res.file ) );

    return res;
}

Expected<void> writeGavHeader( std::ostream & out, const Vector3i & dims, const Vector3f & voxSize, const MinMaxf & mm )
{
    Json::Value headerJson;
    headerJson["ValueType"] = "Float";

    Json::Value dimsJson;
    dimsJson["X"] = dims.x;
    dimsJson["Y"] = dims.y;
    dimsJson["Z"] = dims.z;
    headerJson["Dimensions"] = dimsJson;

    Json::Value voxJson;
    voxJson["X"] = voxSize.x;
    voxJson["Y"] = voxSize.y;
    voxJson["Z"] = voxSize.z;
    headerJson["VoxelSize"] = voxJson;

    Json::Value rangeJson;
    rangeJson["Min"] = mm.min;
    rangeJson["Max"] = mm.max;
    headerJson["Range"] = rangeJson;

    const auto headerRes = serializeJsonValue( headerJson );
    if ( !headerRes )
        return unexpected( "Header composition error" );
    const auto& header = *headerRes;

    const auto headerLen = uint32_t( header.size() );
    out.write( (const char*)&headerLen, sizeof( headerLen ) );
    out.write( header.data(), headerLen );
    if ( !out )
        return unexpected( "Header write error" );
    return {};
}

} // anonymous namespace

Expected<void> toRawAutoname( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    return openRawAutonameStream( vdbVolume.dims, vdbVolume.voxelSize, vdbVolume.data->getGridClass() == openvdb::GRID_LEVEL_SET, file ).and_then(
        [&]( NamedOutFileStream && s )
        {
            return addFileNameInError( toRawFloat( vdbVolume, s.out, callback ), s.file );
        }
    );
}

Expected<void> toRawAutoname( const SimpleVolume& simpleVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    return openRawAutonameStream( simpleVolume.dims, simpleVolume.voxelSize, false, file ).and_then(
        [&]( NamedOutFileStream && s )
        {
            return addFileNameInError( toRawFloat( simpleVolume, s.out, callback ), s.file );
        }
    );
}

Expected<void> gridToRawAutoname( const FloatGrid& grid, const Vector3i& dims, const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;

    return openRawAutonameStream( dims, Vector3f::diagonal( 1.0f ), grid->getGridClass() == openvdb::GRID_LEVEL_SET, file ).and_then(
        [&] ( NamedOutFileStream&& s )
    {
        return addFileNameInError( gridToRawFloat( grid, dims, s.out, callback ), s.file );
    }
    );
}

Expected<void> toGav( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return addFileNameInError( toGav( vdbVolume, out, callback ), file );
}

Expected<void> toGav( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER;
    return writeGavHeader( out, vdbVolume.dims, vdbVolume.voxelSize, { vdbVolume.min, vdbVolume.max } ).and_then(
        [&]()
        {
            return toRawFloat( vdbVolume, out, callback );
        }
    );
}

Expected<void> toGav( const SimpleVolumeMinMax& simpleVolumeMinMax, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return addFileNameInError( toGav( simpleVolumeMinMax, out, callback ), file );
}

Expected<void> toGav( const SimpleVolumeMinMax& simpleVolumeMinMax, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER;
    return writeGavHeader( out, simpleVolumeMinMax.dims, simpleVolumeMinMax.voxelSize, { simpleVolumeMinMax.min, simpleVolumeMinMax.max } ).and_then(
        [&]()
        {
            return toRawFloat( simpleVolumeMinMax, out, callback );
        }
    );
}

Expected<void> toGav( const SimpleVolume& simpleVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return addFileNameInError( toGav( simpleVolume, out, callback ), file );
}

Expected<void> toGav( const SimpleVolume& simpleVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER;
    auto [min, max] = parallelMinMax( simpleVolume.data );
    return writeGavHeader( out, simpleVolume.dims, simpleVolume.voxelSize, { min, max } ).and_then(
        [&]()
        {
            return toRawFloat( simpleVolume, out, callback );
        }
    );
}

Expected<void> toVdb( const VdbVolume& vdbVolume, const std::filesystem::path& filename, ProgressCallback callback )
{
    MR_TIMER;
    FloatGrid newGrid = std::make_shared<OpenVdbFloatGrid>();
    newGrid->setTree( vdbVolume.data->treePtr() );
    newGrid->setGridClass( vdbVolume.data->getGridClass() );
    openvdb::math::Transform::Ptr transform = std::make_shared<openvdb::math::Transform>();
    transform->preScale( { vdbVolume.voxelSize.x, vdbVolume.voxelSize.y, vdbVolume.voxelSize.z } );
    newGrid->setTransform( transform );

    // in order to save on Windows a file with Unicode symbols in the name, we need to open ofstream by ourselves,
    // because openvdb constructs it from std::string, which on Windows means "local codepage" and not Unicode
    return gridToVdb( newGrid, filename, callback );
}

MR_FORMAT_REGISTRY_IMPL( VoxelsSaver )


Expected<void> gridToVdb( const FloatGrid& grid, const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return addFileNameInError( gridToVdb( grid, out, callback ), file );
}

Expected<void> gridToVdb( const FloatGrid& vdbVolume, std::ostream& out, ProgressCallback /*callback*/ /*= {} */ )
{
    openvdb::io::Stream stream( out );
    stream.write( openvdb::GridCPtrVec{ vdbVolume.toVdb() } );
    if ( !out )
        return unexpected( "error writing in stream" );
    return {};
}

Expected<void> toAnySupportedFormat( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                   ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );
    ext = "*" + ext;

    auto saver = getVoxelsSaver( ext );
    if ( !saver )
        return unexpectedUnsupportedFileExtension();

    return saver( vdbVolume, file, callback );
}

Expected<void> gridToAnySupportedFormat( const FloatGrid& grid, const Vector3i& dims, const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext == ".raw" )
        return gridToRawAutoname( grid, dims, file, callback );
    else if ( ext == ".vdb" )
        return gridToVdb( grid, file, callback );

    return unexpectedUnsupportedFileExtension();
}

template <VoxelsSaver voxelsSaver>
Expected<void> toVoxels( const Object& object, const std::filesystem::path& path, const ProgressCallback& callback )
{
    const auto objVoxels = getAllObjectsInTree<ObjectVoxels>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable );
    if ( objVoxels.empty() )
        return voxelsSaver( {}, path, callback );
    else if ( objVoxels.size() > 1 )
        return unexpected( "Multiple voxel grids in the given object" );

    const auto& objVoxel = objVoxels.front();
    if ( !objVoxel )
        return voxelsSaver( {}, path, callback );

    return voxelsSaver( objVoxel->vdbVolume(), path, callback );
}

#define MR_ADD_VOXELS_SAVER( filter, saver )                   \
MR_ON_INIT {                                                   \
    MR::VoxelsSave::setVoxelsSaver( filter, saver );           \
    /* additionally register the saver as an object saver */   \
    MR::ObjectSave::setObjectSaver( filter, toVoxels<saver> ); \
};

MR_ADD_VOXELS_SAVER( IOFilter( "Raw (.raw)", "*.raw" ), toRawAutoname )
MR_ADD_VOXELS_SAVER( IOFilter( "Micro CT (.gav)", "*.gav" ), toGav )
MR_ADD_VOXELS_SAVER( IOFilter( "OpenVDB (.vdb)", "*.vdb" ), toVdb )

Expected<void> saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlane& slicePlain, int sliceNumber, ProgressCallback callback )
{
    const auto& dims = vdbVolume.dims;
    const int textureWidth = dims[( slicePlain + 1 ) % 3];
    const int textureHeight = dims[( slicePlain + 2 ) % 3];

    std::vector<Color> texture( textureWidth * textureHeight );
    Vector3i activeVoxel;
    switch ( slicePlain )
    {
    case SlicePlane::XY:
        if ( sliceNumber > dims.z )
            return unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { 0, 0, sliceNumber };
        break;
    case SlicePlane::YZ:
        if ( sliceNumber > dims.x )
            return unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { sliceNumber, 0, 0 };
        break;
    case SlicePlane::ZX:
        if ( sliceNumber > dims.y )
            return unexpected( "Slice number exceeds voxel object borders" );

        activeVoxel = { 0, sliceNumber, 0 };
        break;
    default:
        return unexpected( "Slice plain is invalid" );
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
            return unexpectedOperationCanceled();
    }

    MeshTexture meshTexture( { { std::move( texture ), {textureWidth, textureHeight} } } );
    auto saveRes = ImageSave::toAnySupportedFormat( meshTexture, path );
    if ( !saveRes.has_value() )
        return unexpected( saveRes.error() );

    if ( callback )
        callback( 1.0f );

    return {};
}

Expected<void> saveAllSlicesToImage( const VdbVolume& vdbVolume, const SavingSettings& settings )
{
    int numSlices{ 0 };
    switch ( settings.slicePlane )
    {
    case SlicePlane::XY:
        numSlices = vdbVolume.dims.z;
        break;
    case SlicePlane::YZ:
        numSlices = vdbVolume.dims.x;
        break;
    case SlicePlane::ZX:
        numSlices = vdbVolume.dims.y;
        break;
    default:
        return unexpected( "Slice plane is invalid" );
    }

    const size_t maxNumChars = std::to_string( numSlices ).size();
    for ( int i = 0; i < numSlices; ++i )
    {
        const auto res = saveSliceToImage( settings.path / fmt::format( runtimeFmt( settings.format ), i, maxNumChars ), vdbVolume, settings.slicePlane, i );
        if ( !res )
            return res;

        if ( settings.cb && !settings.cb( float( i ) / numSlices ) )
            return unexpectedOperationCanceled();
    }

    if ( settings.cb )
        settings.cb( 1.f );
    return {};
}

} // namespace VoxelsSave

Expected<void> saveObjectVoxelsToFile( const Object& object, const std::filesystem::path& path, const ProgressCallback& callback )
{
    return VoxelsSave::toVoxels<VoxelsSave::toAnySupportedFormat>( object, path, callback );
}

} // namespace MR
