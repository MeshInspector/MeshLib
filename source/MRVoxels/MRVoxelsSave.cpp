#include "MRVoxelsSave.h"

#include "MRObjectVoxels.h"
#include "MRMesh/MRImageSave.h"
#include "MRVDBFloatGrid.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRProgressReadWrite.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRTimer.h"
#include "MROpenVDB.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRVDBConversions.h"

#include <openvdb/io/Stream.h>

#pragma warning(push)
#pragma warning(disable: 4515)
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <gdcmImageWriter.h>
#pragma warning(pop)

#include <fstream>
#include <filesystem>
#include <sstream>

namespace MR
{

namespace VoxelsSave
{

VoidOrErrStr toRawFloat( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER
    const auto& grid = vdbVolume.data;
    auto accessor = grid->getConstAccessor();
    const auto& dims = vdbVolume.dims;

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

    if ( !writeByBlocks( out, (const char*) buffer.data(), buffer.size() * sizeof( float ), callback ) )
        return unexpected( std::string( "Saving canceled" ) );
    if ( !out )
        return unexpected( std::string( "Stream write error" ) );

    return {};
}

VoidOrErrStr toRawAutoname( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER
    if ( file.empty() )
    {
        return unexpected( "Filename is empty" );
    }

    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return unexpected( ss.str() );
    }

    const auto& dims = vdbVolume.dims;
    if ( dims.x == 0 || dims.y == 0 || dims.z == 0 )
    {
        return unexpected( "VdbVolume is empty" );
    }

    std::stringstream prefix;
    prefix.precision( 3 );
    prefix << "W" << dims.x << "_H" << dims.y << "_S" << dims.z;    // dims
    const auto& voxSize = vdbVolume.voxelSize;
    prefix << "_V" << voxSize.x * 1000.0f << "_" << voxSize.y * 1000.0f << "_" << voxSize.z * 1000.0f; // voxel size "_F" for float
    prefix << "_G" << ( vdbVolume.data->getGridClass() == openvdb::GRID_LEVEL_SET ? "1" : "0" ) << "_F ";
    prefix << utf8string( file.filename() );                        // name

    std::filesystem::path outPath = file;
    outPath.replace_filename( prefix.str() );
    std::ofstream out( outPath, std::ios::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( outPath ) );

    return addFileNameInError( toRawFloat( vdbVolume, out, callback ), outPath );
}

VoidOrErrStr toGav( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return addFileNameInError( toGav( vdbVolume, out, callback ), file );
}

VoidOrErrStr toGav( const VdbVolume& vdbVolume, std::ostream & out, ProgressCallback callback )
{
    MR_TIMER
    Json::Value headerJson;
    headerJson["ValueType"] = "Float";

    Json::Value dimsJson;
    dimsJson["X"] = vdbVolume.dims.x;
    dimsJson["Y"] = vdbVolume.dims.y;
    dimsJson["Z"] = vdbVolume.dims.z;
    headerJson["Dimensions"] = dimsJson;

    Json::Value voxJson;
    voxJson["X"] = vdbVolume.voxelSize.x;
    voxJson["Y"] = vdbVolume.voxelSize.y;
    voxJson["Z"] = vdbVolume.voxelSize.z;
    headerJson["VoxelSize"] = voxJson;

    Json::Value rangeJson;
    rangeJson["Min"] = vdbVolume.min;
    rangeJson["Max"] = vdbVolume.max;
    headerJson["Range"] = rangeJson;

    std::ostringstream oss;
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( writer->write( headerJson, &oss ) != 0 || !oss )
        return unexpected( "Header composition error" );

    const auto header = oss.str();
    const auto headerLen = uint32_t( header.size() );
    out.write( (const char*)&headerLen, sizeof( headerLen ) );
    out.write( header.data(), headerLen );
    if ( !out )
        return unexpected( "Header write error" );
    
    return toRawFloat( vdbVolume, out, callback );
}

VoidOrErrStr toVdb( const VdbVolume& vdbVolume, const std::filesystem::path& filename, ProgressCallback /*callback*/ )
{
    MR_TIMER
    openvdb::FloatGrid::Ptr gridPtr = std::make_shared<openvdb::FloatGrid>();
    gridPtr->setTree( vdbVolume.data->treePtr() );
    gridPtr->setGridClass( vdbVolume.data->getGridClass() );
    openvdb::math::Transform::Ptr transform = std::make_shared<openvdb::math::Transform>();
    transform->preScale( { vdbVolume.voxelSize.x, vdbVolume.voxelSize.y, vdbVolume.voxelSize.z } );
    gridPtr->setTransform( transform );

    // in order to save on Windows a file with Unicode symbols in the name, we need to open ofstream by ourselves,
    // because openvdb constructs it from std::string, which on Windows means "local codepage" and not Unicode
    std::ofstream file( filename, std::ios::binary );
    if ( !file )
        return unexpected( "cannot open file for writing: " + utf8string( filename ) );

    openvdb::io::Stream stream( file );
    stream.write( openvdb::GridCPtrVec{ gridPtr } );
    if ( !file )
        return unexpected( "error writing in file: " + utf8string( filename ) );

    return {};
}


template <typename T>
std::pair<gdcm::PixelFormat::ScalarType, gdcm::Tag> getGDCMTypeAndTag()
{
    // https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.24.html
    // https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html
    // https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.25.html

    if constexpr ( std::floating_point<T> )
        static_assert( dependent_false<T>, "GDCM doesn't support floating point DICOMs" );
    else if constexpr ( std::same_as<T, uint16_t> )
        return { gdcm::PixelFormat::ScalarType::UINT16, gdcm::Tag( 0x7FE0, 0x0010 ) };
    else
        static_assert( dependent_false<T>, "Unsupported type T" );
}


VoidOrErrStr toDCM( const VdbVolume& vdbVolume, const std::filesystem::path& path, ProgressCallback cb )
{
    auto simpleVolume = vdbVolumeToSimpleVolumeU16( vdbVolume, {}, subprogress( cb, 0.f, 0.5f ) );
    if ( simpleVolume )
        return toDCM( *simpleVolume, path, subprogress( cb, 0.5f, 1.f ) );
    else
        return unexpected( simpleVolume.error() );
}

template <typename T>
VoidOrErrStr toDCM( const VoxelsVolume<std::vector<T>>& volume, const std::filesystem::path& path, ProgressCallback cb )
{
    if ( !reportProgress( cb, 0.0f ) )
        return unexpected( "Loading canceled" );

    auto [gdcmScalar, gdcmTag] = getGDCMTypeAndTag<T>();

    gdcm::ImageWriter iw;
    auto& image = iw.GetImage();
    image.SetNumberOfDimensions( 3 );
    image.SetDimension( 0, volume.dims.x );
    image.SetDimension( 1, volume.dims.y );
    image.SetDimension( 2, volume.dims.z );
    image.SetPixelFormat( gdcm::PixelFormat( gdcmScalar ) );
    image.SetPhotometricInterpretation( gdcm::PhotometricInterpretation::MONOCHROME2 );
    image.SetSpacing( 0, volume.voxelSize.x * 1000.f );
    image.SetSpacing( 1, volume.voxelSize.y * 1000.f );
    image.SetSpacing( 2, volume.voxelSize.z * 1000.f );

    gdcm::DataElement data( gdcmTag );
    // copies full volume
    data.SetByteValue( reinterpret_cast<const char*>( volume.data.data() ), ( uint32_t )volume.data.size() * sizeof( T ) );
    if ( !reportProgress( cb, 0.5f ) )
        return unexpected( "Loading canceled" );
    image.SetDataElement( data );

    iw.SetImage( image );

    std::ofstream fout( path );
    iw.SetStream( fout );
    if ( !fout || !iw.Write() )
        return unexpected( "Cannot write DICOM file" );

    return {};
}

template VoidOrErrStr toDCM<uint16_t>( const SimpleVolumeU16& volume, const std::filesystem::path& path, ProgressCallback cb );


MR_FORMAT_REGISTRY_IMPL( VoxelsSaver )

VoidOrErrStr toAnySupportedFormat( const VdbVolume& vdbVolume, const std::filesystem::path& file,
                                   ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );
    ext = "*" + ext;

    auto saver = getVoxelsSaver( ext );
    if ( !saver )
        return unexpected( std::string( "unsupported file extension" ) );

    return saver( vdbVolume, file, callback );
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
MR_ADD_VOXELS_SAVER( IOFilter( "Dicom (.dcm)", "*.dcm" ), toDCM )

VoidOrErrStr saveSliceToImage( const std::filesystem::path& path, const VdbVolume& vdbVolume, const SlicePlane& slicePlain, int sliceNumber, ProgressCallback callback )
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

VoidOrErrStr saveAllSlicesToImage( const VdbVolume& vdbVolume, const SavingSettings& settings )
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

Expected<void> saveObjectVoxelsToFile( const Object& object, const std::filesystem::path& path, ProgressCallback callback )
{
    return VoxelsSave::toVoxels<VoxelsSave::toAnySupportedFormat>( object, path, callback );
}

} // namespace MR
