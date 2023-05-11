#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRVoxelsLoad.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRObjectVoxels.h"
#include "MRVDBConversions.h"
#include "MRStringConvert.h"
#include "MRFloatGrid.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include "MRStringConvert.h"
#include <compare>
#include <filesystem>
#include <fstream>

#ifndef MRMESH_NO_DICOM
#include <gdcmImageHelper.h>
#include <gdcmImageReader.h>
#include <gdcmTagKeywords.h>
#endif // MRMESH_NO_DICOM

#include <MRPch/MROpenvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>
#include <MRPch/MRTBB.h>
#ifndef MRMESH_NO_TIFF
#include <tiffio.h>
#endif // MRMESH_NO_TIFF

#include "MROpenVDBHelper.h"
namespace
{
    using namespace MR::VoxelsLoad;

#ifndef MRMESH_NO_DICOM
    RawParameters::ScalarType convertToScalarType( const gdcm::PixelFormat& format )
    {
        switch ( gdcm::PixelFormat::ScalarType( format ) )
        {
        case gdcm::PixelFormat::UINT8:
            return RawParameters::ScalarType::UInt8;
        case gdcm::PixelFormat::INT8:
            return RawParameters::ScalarType::Int8;
        case gdcm::PixelFormat::UINT16:
            return RawParameters::ScalarType::UInt16;
        case gdcm::PixelFormat::INT16:
            return RawParameters::ScalarType::Int16;
        case gdcm::PixelFormat::UINT32:
            return RawParameters::ScalarType::UInt32;
        case gdcm::PixelFormat::INT32:
            return RawParameters::ScalarType::Int32;
        case gdcm::PixelFormat::UINT64:
            return RawParameters::ScalarType::UInt64;
        case gdcm::PixelFormat::INT64:
            return RawParameters::ScalarType::Int64;
        default:
            return RawParameters::ScalarType::Unknown;
        }
    }
#endif // MRMESH_NO_DICOM
}

namespace MR
{

namespace VoxelsLoad
{

const IOFilters Filters =
{
    {"Raw (.raw)","*.raw"},
    {"OpenVDB (.vdb)","*.vdb"},
    {"Micro CT (.gav)","*.gav"},
};

struct SliceInfoBase
{
    int instanceNum = 0;
    double z = 0;
    int fileNum = 0;
    auto operator <=>( const SliceInfoBase & ) const = default;
};

struct SliceInfo : SliceInfoBase
{
    // these fields will be ignored in sorting
    Vector3d imagePos;
};

void sortByOrder( std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder )
{
    std::sort( zOrder.begin(), zOrder.end() );
    auto filesSorted = scans;
    for ( int i = 0; i < scans.size(); ++i )
        filesSorted[i] = scans[zOrder[i].fileNum];
    scans = std::move( filesSorted );
}

void putFileNameInZ( const std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder )
{
    assert( zOrder.size() == scans.size() );
    tbb::parallel_for( tbb::blocked_range( 0, int( scans.size() ) ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            std::string name = utf8string( scans[i].stem() );
            auto pos = name.find_first_of( "-0123456789" );
            double res = 0.0;
            if ( pos != std::string::npos )
            {
                auto subName = name.substr( pos );
                res = std::atof( subName.c_str() );
            }
            assert( zOrder[i].fileNum == i );
            zOrder[i].z = res;
        }
    } );
}

void sortFilesByName( std::vector<std::filesystem::path>& scans )
{
    const auto sz = scans.size();
    std::vector<SliceInfo> zOrder( sz );
    for ( int i = 0; i < sz; ++i )
        zOrder[i].fileNum = i;
    putFileNameInZ( scans, zOrder );
    sortByOrder( scans, zOrder );
}

std::function<float( char* )> getTypeConverter( const RawParameters::ScalarType& scalarType, const uint64_t& range, const int64_t& min )
{
    switch ( scalarType )
    {
    case RawParameters::ScalarType::UInt8:
        return [range, min]( char* c )
        {
            return float( *(uint8_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::UInt16:
        return [range, min]( char* c )
        {
            return float( *(uint16_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::Int8:
        return [range, min]( char* c )
        {
            return float( *(int8_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::Int16:
        return [range, min]( char* c )
        {
            return float( *(int16_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::Int32:
        return [range, min]( char* c )
        {
            return float( *(int32_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::UInt32:
        return [range, min]( char* c )
        {
            return float( *(uint32_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::UInt64:
        return [range, min]( char* c )
        {
            return float( *(uint64_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::Int64:
        return [range, min]( char* c )
        {
            return float( *(int64_t*) (c) -min ) / float( range );
        };
    case RawParameters::ScalarType::Float32:
        return []( char* c )
        {
            return *(float*) ( c );
        };
    case RawParameters::ScalarType::Float64:
        return []( char* c )
        {
            return float( *(double*) ( c ) );
        };
    case RawParameters::ScalarType::Unknown:
    case RawParameters::ScalarType::Count:
        break;
    }
    return {};
}

#ifndef MRMESH_NO_DICOM
bool isDICOMFile( const std::filesystem::path& path )
{
    gdcm::ImageReader ir;
    std::ifstream ifs( path, std::ios_base::binary );
    ir.SetStream( ifs );
    if ( !ir.CanRead() )
        return false;
    // we read these tags to be able to determine whether this file is dicom dir or image
    auto tags = {
        gdcm::Tag( 0x0002, 0x0002 ), // media storage
        gdcm::Tag( 0x0008, 0x0016 ), // media storage
        gdcm::Keywords::PhotometricInterpretation::GetTag(),
        gdcm::Keywords::ImagePositionPatient::GetTag(), // is for image origin in mm
        gdcm::Tag( 0x0028, 0x0010 ),gdcm::Tag( 0x0028, 0x0011 ),gdcm::Tag( 0x0028, 0x0008 )}; // is for dimensions
    if ( !ir.ReadSelectedTags( tags ) )
        return false;
    gdcm::MediaStorage ms;
    ms.SetFromFile( ir.GetFile() );

    // skip unsupported media storage
    if ( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage || ms == gdcm::MediaStorage::SecondaryCaptureImageStorage
        || ms == gdcm::MediaStorage::BasicTextSR )
    {
        spdlog::warn( "DICOM file {} has unsupported media storage {}", utf8string( path ), (int)ms );
        return false;
    }

    // unfortunatly gdcm::ImageHelper::GetPhotometricInterpretationValue returns something even if no data in the file
    if ( !gdcm::ImageHelper::GetPointerFromElement( gdcm::Keywords::PhotometricInterpretation::GetTag(), ir.GetFile() ) )
    {
        spdlog::warn( "DICOM file {} does not have Photometric Interpretation", utf8string( path ) );
        return false;
    }

    auto photometric = gdcm::ImageHelper::GetPhotometricInterpretationValue( ir.GetFile() );
    if ( photometric != gdcm::PhotometricInterpretation::MONOCHROME2 &&
         photometric != gdcm::PhotometricInterpretation::MONOCHROME1 )
    {
        spdlog::warn( "DICOM file {} has Photometric Interpretation other than Monochrome", utf8string( path ) );
        return false;
    }

    auto dims = gdcm::ImageHelper::GetDimensionsValue( ir.GetFile() );
    if ( dims.size() != 3 )
    {
        spdlog::warn( "DICOM file {} has Dimensions Value other than 3", utf8string( path ) );
        return false;
    }

    return true;
}

struct DCMFileLoadResult
{
    bool success = false;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    std::string seriesDescription;
    AffineXf3f xf;
};

DCMFileLoadResult loadSingleFile( const std::filesystem::path& path, SimpleVolume& data, size_t offset )
{
    MR_TIMER;
    DCMFileLoadResult res;

    std::ifstream fstr( path, std::ifstream::binary );
    gdcm::ImageReader ir;
    ir.SetStream( fstr );

    if ( !ir.Read() )
    {
        spdlog::error( "Cannot read image from DICOM file {}", utf8string( path ) );
        return res;
    }

    const gdcm::DataSet& ds = ir.GetFile().GetDataSet();
    if( ds.FindDataElement( gdcm::Keywords::SeriesDescription::GetTag() ) )
    {
        const gdcm::DataElement& de = ds.GetDataElement( gdcm::Keywords::SeriesDescription::GetTag() );
        gdcm::Keywords::SeriesDescription desc;
        desc.SetFromDataElement( de );
        auto descVal = desc.GetValue();
        res.seriesDescription = descVal;
    }

    if( ds.FindDataElement( gdcm::Keywords::ImagePositionPatient::GetTag() ) )
    {
        gdcm::DataElement dePosition = ds.GetDataElement( gdcm::Keywords::ImagePositionPatient::GetTag() );
        gdcm::Keywords::ImagePositionPatient atPos;
        atPos.SetFromDataElement( dePosition );
        for (int i = 0; i < 3; ++i) {
            res.xf.b[i] = float( atPos.GetValue( i ) );
        }
        res.xf.b /= 1000.0f;
    }

    if( ds.FindDataElement( gdcm::Keywords::ImageOrientationPatient::GetTag() ) )
    {
        gdcm::DataElement deOri = ds.GetDataElement( gdcm::Keywords::ImageOrientationPatient::GetTag() );
        gdcm::Keywords::ImageOrientationPatient atOri;
        atOri.SetFromDataElement( deOri );
        for (int i = 0; i < 3; ++i)
            res.xf.A.x[i] = float( atOri.GetValue( i ) );
        for (int i = 0; i < 3; ++i)
            res.xf.A.y[i] = float( atOri.GetValue( 3 + i ) );
    }

    res.xf.A.x.normalized();
    res.xf.A.y.normalized();
    res.xf.A.z = cross( res.xf.A.x, res.xf.A.y );
    res.xf.A = res.xf.A.transposed();

    const auto& gimage = ir.GetImage();
    auto dimsNum = gimage.GetNumberOfDimensions();
    const unsigned* dims = gimage.GetDimensions();
    bool needInvertZ = false;

    if ( data.dims.x == 0 || data.dims.y == 0 )
    {
        data.dims.x = dims[0];
        data.dims.y = dims[1];
    }
    if ( dimsNum == 3 )
    {
        data.dims.z = dims[2];
    }
    if ( data.voxelSize[0] == 0.0f )
    {
        const double* spacing = gimage.GetSpacing();
        if ( spacing[0] == 1 && spacing[1] == 1 && spacing[2] == 1 )
        {
            // gdcm was unable to find the spacing, so find it by ourselves
            if( ds.FindDataElement( gdcm::Keywords::PixelSpacing::GetTag() ) )
            {
                const gdcm::DataElement& de = ds.GetDataElement( gdcm::Keywords::PixelSpacing::GetTag() );
                gdcm::Keywords::PixelSpacing desc;
                desc.SetFromDataElement( de );
                data.voxelSize.x = float( desc.GetValue(0) / 1000 );
                data.voxelSize.y = float( desc.GetValue(1) / 1000 );
            }
        }
        else 
        {
            data.voxelSize.x = float( spacing[0] / 1000 );
            data.voxelSize.y = float( spacing[1] / 1000 );
        }
        if ( data.voxelSize.z == 0.0f )
        {
            if ( dimsNum == 3 )
            {
                float spacingZ = 0.0f;
                if ( ds.FindDataElement( gdcm::Keywords::SpacingBetweenSlices::GetTag() ) )
                {
                    const gdcm::DataElement& de = ds.GetDataElement( gdcm::Keywords::SpacingBetweenSlices::GetTag() );
                    gdcm::Keywords::SpacingBetweenSlices desc;
                    desc.SetFromDataElement( de );
                    spacingZ = float( desc.GetValue() );
                    // looks like if this tag is set image stored inverted by Z
                    // no other tags was found to determine orientation (compared with cases without this tag)
                    needInvertZ = spacingZ > 0.0f;
                }
                else
                {
                    spacingZ = float( spacing[2] );
                    needInvertZ = spacingZ < 0.0f;
                }
                data.voxelSize.z = std::abs( spacingZ ) * 1e-3f;
            }
            else
                data.voxelSize.z = data.voxelSize.x;
        }
    }
    else if ( data.dims.x != (int) dims[0] || data.dims.y != (int) dims[1] )
    {
        spdlog::error( "loadSingle: dimensions are inconsistent with other files, file: {}", utf8string( path ) );
        return res;
    }
    if ( gimage.GetPhotometricInterpretation() != gdcm::PhotometricInterpretation::MONOCHROME2 &&
         gimage.GetPhotometricInterpretation() != gdcm::PhotometricInterpretation::MONOCHROME1 )
    {
        spdlog::error( "loadSingle: unexpected PhotometricInterpretation, file: {}", utf8string( path ) );
        spdlog::error( "PhotometricInterpretation: {}", gimage.GetPhotometricInterpretation() );
        return res;
    }
    auto min = gimage.GetPixelFormat().GetMin();
    auto max = gimage.GetPixelFormat().GetMax();
    auto pixelSize = gimage.GetPixelFormat().GetPixelSize();
    auto scalarType = convertToScalarType( gimage.GetPixelFormat() );
    auto caster = getTypeConverter( scalarType, max - min, min );
    if ( !caster )
    {
        spdlog::error( "loadSingle: cannot make type converter, file: {}", utf8string( path ) );
        spdlog::error( "Type: {}", gimage.GetPixelFormat() );
        return res;
    }
    std::vector<char> cacheBuffer( gimage.GetBufferLength() );
    if ( !gimage.GetBuffer( cacheBuffer.data() ) )
    {
        spdlog::error( "loadSingle: cannot load data from file: {}", utf8string( path ) );
        return res;
    }
    size_t fulSize = size_t( data.dims.x )*data.dims.y*data.dims.z;
    if ( data.data.size() != fulSize )
        data.data.resize( fulSize );

    auto dimZ = dimsNum == 3 ? dims[2] : 1;
    auto dimXY = dims[0] * dims[1];
    auto dimXYZinv = dimZ * dimXY - dimXY;
    for ( unsigned z = 0; z < dimZ; ++z )
    {
        auto zOffset = z * dimXY;
        auto correctZOffset = needInvertZ ? ( dimXYZinv - zOffset ) : zOffset;
        for ( size_t i = 0; i < dimXY; ++i )
        {
            auto f = caster( &cacheBuffer[( correctZOffset + i ) * pixelSize] );
            res.min = std::min( res.min, f );
            res.max = std::max( res.max, f );
            data.data[zOffset + offset + i] = f;
        }
    }
    res.success = true;

    return res;
}

void sortDICOMFiles( std::vector<std::filesystem::path>& files, unsigned maxNumThreads, Vector3f& voxelSize )
{
    if ( files.empty() )
        return;

    std::vector<SliceInfo> zOrder( files.size() );

    tbb::task_arena limitedArena( maxNumThreads );
    limitedArena.execute( [&]
    {
        tbb::parallel_for( tbb::blocked_range( 0, int( files.size() ) ),
            [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                gdcm::ImageReader ir;
                std::ifstream ifs( files[i], std::ios_base::binary );
                ir.SetStream( ifs );
                ir.ReadSelectedTags( { 
                    gdcm::Tag( 0x0002, 0x0002 ),
                    gdcm::Tag( 0x0008, 0x0016 ),
                    gdcm::Keywords::InstanceNumber::GetTag(),
                    gdcm::Keywords::ImagePositionPatient::GetTag() } );

                SliceInfo sl;
                sl.fileNum = i;
                const auto origin = gdcm::ImageHelper::GetOriginValue( ir.GetFile() );
                sl.z = origin[2];
                sl.imagePos = { origin[0], origin[1], origin[2] };

                // if Instance Number is available then sort by it
                const gdcm::DataSet& ds = ir.GetFile().GetDataSet();
                if( ds.FindDataElement( gdcm::Keywords::InstanceNumber::GetTag() ) )
                {
                    const gdcm::DataElement& de = ds.GetDataElement( gdcm::Keywords::InstanceNumber::GetTag() );
                    gdcm::Keywords::InstanceNumber at = {0}; // default value if empty
                    at.SetFromDataElement( de );
                    sl.instanceNum = at.GetValue();
                }
                zOrder[i] = sl;
            }
        } );
    } );

    bool zPosPresent = std::any_of( zOrder.begin(), zOrder.end(), [] ( const SliceInfo& el )
    {
        return el.z != 0.0;
    } );
    if ( !zPosPresent )
    {
        putFileNameInZ( files, zOrder );
    }

    sortByOrder( files, zOrder );
    if ( zOrder.size() > 1 )
    {
        voxelSize.z = float( ( zOrder[1].imagePos - zOrder[0].imagePos ).length() / 1000.0 );
        // if slices go in descending z-order then reverse them
        if ( zOrder[1].imagePos.z < zOrder[0].imagePos.z )
            std::reverse( files.begin(), files.end() );
    }
}

tl::expected<DicomVolume, std::string> loadDicomFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads, const ProgressCallback& cb )
{
    MR_TIMER
    if ( !reportProgress( cb, 0.0f ) )
        return tl::make_unexpected( "Loading canceled" );

    SimpleVolume data;
    data.voxelSize = Vector3f();
    data.dims = Vector3i::diagonal( 0 );
    std::error_code ec;
    if ( !std::filesystem::is_directory( path, ec ) )
        return tl::make_unexpected( "loadDCMFolder: path is not directory" );

    int filesNum = 0;
    std::vector<std::filesystem::path> files;
    const std::filesystem::directory_iterator dirEnd;
    for ( auto it = std::filesystem::directory_iterator( path, ec ); !ec && it != dirEnd; it.increment( ec ) )
    {
        if ( it->is_regular_file( ec ) )
            ++filesNum;
    }
    files.reserve( filesNum );
    int fCounter = 0;
    for ( auto it = std::filesystem::directory_iterator( path, ec ); !ec && it != dirEnd; it.increment( ec ) )
    {
        ++fCounter;
        auto filePath = it->path();
        if ( it->is_regular_file( ec ) && isDICOMFile( filePath ) )
            files.push_back( filePath );
        if ( !reportProgress( cb, 0.3f * float( fCounter ) / float( filesNum ) ) )
            tl::make_unexpected( "Loading canceled" );
    }
    if ( files.empty() )
    {
        return tl::make_unexpected( "loadDCMFolder: there is no dcm file in folder: " + utf8string( path ) );
    }
    if ( files.size() == 1 )
        return loadDicomFile( files[0], subprogress( cb, 0.3f, 1.0f ) );
    sortDICOMFiles( files, maxNumThreads, data.voxelSize );
    data.dims.z = (int) files.size();

    auto firstRes = loadSingleFile( files.front(), data, 0 );
    if ( !firstRes.success )
        return tl::make_unexpected( "loadDCMFolder: error loading first file \"" + utf8string( files.front() ) + "\"" );
    data.min = firstRes.min;
    data.max = firstRes.max;
    size_t dimXY = data.dims.x * data.dims.y;

    if ( !reportProgress( cb, 0.4f ) )
        return tl::make_unexpected( "Loading canceled" );

    // other slices
    auto mainThreadId = std::this_thread::get_id();
    bool cancelCalled = false;
    std::vector<DCMFileLoadResult> slicesRes( files.size() - 1 );
    tbb::task_arena limitedArena( maxNumThreads );
    std::atomic<int> numLoadedSlices = 0;
    limitedArena.execute( [&]
    {
        tbb::parallel_for( tbb::blocked_range( 0, int( slicesRes.size() ) ),
                           [&]( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                slicesRes[i] = loadSingleFile( files[i + 1], data, ( i + 1 ) * dimXY );
                ++numLoadedSlices;
                if ( cb && std::this_thread::get_id() == mainThreadId )
                    cancelCalled = !cb( 0.4f + 0.6f * ( float( numLoadedSlices ) / float( slicesRes.size() ) ) );
            }
        } );
    } );
    if ( cancelCalled )
        return tl::make_unexpected( "Loading canceled" );

    for ( const auto& sliceRes : slicesRes )
    {
        if ( !sliceRes.success )
            return {};
        data.min = std::min( sliceRes.min, data.min );
        data.max = std::max( sliceRes.max, data.max );
    }

    DicomVolume res;
    res.vol = std::move( data );
    if ( firstRes.seriesDescription.empty() )
         res.name = utf8string( files.front().parent_path().stem() );
    else
         res.name = firstRes.seriesDescription;
    res.xf = firstRes.xf;
    return res;
}

tl::expected<LoadDCMResult, std::string> loadDCMFolder( const std::filesystem::path& path,
                                                     unsigned maxNumThreads, const ProgressCallback& cb )
{
    auto simple = loadDicomFolder( path, maxNumThreads, subprogress( cb, 0, 0.5f ) );
    if ( !simple.has_value() )
        return tl::make_unexpected( std::move( simple.error() ) );

    LoadDCMResult res;
    res.vdbVolume = simpleVolumeToVdbVolume( simple->vol, subprogress( cb, 0.5f, 1.0f ) );
    res.name = std::move( simple->name );
    res.xf = std::move( simple->xf );

    return res;
}

std::vector<tl::expected<LoadDCMResult, std::string>> loadDCMFolderTree( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb )
{
    MR_TIMER;
    std::vector<tl::expected<LoadDCMResult, std::string>> res;
    auto tryLoadDir = [&]( const std::filesystem::path& dir )
    {
        res.push_back( loadDCMFolder( dir, maxNumThreads, cb ) );
        if ( !res.back().has_value() && res.back().error() == "Loading canceled" )
            return false;
        return true;
    };
    if ( !tryLoadDir( path ) )
        return { tl::make_unexpected( "Loading canceled" ) };

    const std::filesystem::recursive_directory_iterator dirEnd;
    std::error_code ec;
    for ( auto it = std::filesystem::recursive_directory_iterator( path, ec ); !ec && it != dirEnd; it.increment( ec ) )
    {
        if ( it->is_directory( ec ) && !tryLoadDir( *it ) )
            break;
    }
    return res;
}

tl::expected<DicomVolume, std::string> loadDicomFile( const std::filesystem::path& path, const ProgressCallback& cb )
{
    MR_TIMER
    if ( !reportProgress( cb, 0.0f ) )
        return tl::make_unexpected( "Loading canceled" );

    SimpleVolume simpleVolume;
    simpleVolume.voxelSize = Vector3f();
    simpleVolume.dims.z = 1;
    auto fileRes = loadSingleFile( path, simpleVolume, 0 );
    if ( !fileRes.success )
        return tl::make_unexpected( "loadDCMFile: error load file: " + utf8string( path ) );
    simpleVolume.max = fileRes.max;
    simpleVolume.min = fileRes.min;
    
    DicomVolume res;
    res.vol = std::move( simpleVolume );
    res.name = utf8string( path.stem() );
    return res;
}

#endif // MRMESH_NO_DICOM

tl::expected<VdbVolume, std::string> fromRaw( const std::filesystem::path& path,
    const ProgressCallback& cb )
{
    MR_TIMER;

    if ( path.empty() )
    {
        return tl::make_unexpected( "Path is empty" );
    }

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return tl::make_unexpected( ss.str() );
    }

    auto parentPath = path.parent_path();
    std::error_code ec;
    if ( !std::filesystem::is_directory( parentPath, ec ) )
        return tl::make_unexpected( utf8string( parentPath ) + " - is not directory" );
    std::vector<std::filesystem::path> candidatePaths;
    for ( auto entry : std::filesystem::directory_iterator( parentPath, ec ) )
    {
        auto filename = entry.path().filename();
        auto pos = utf8string( filename ).find( utf8string( path.filename() ) );
        if ( pos != std::string::npos )
            candidatePaths.push_back( entry.path() );
    }
    if ( candidatePaths.empty() )
        return tl::make_unexpected( "Cannot find file: " + utf8string( path.filename() ) );
    else if ( candidatePaths.size() > 1 )
        return tl::make_unexpected( "More than one file exists: " + utf8string( path.filename() ) );

    RawParameters outParams;
    auto filepathToOpen = candidatePaths[0];
    auto filename = utf8string( filepathToOpen.filename() );
    auto wEndChar = filename.find("_");
    if ( wEndChar == std::string::npos )
        return tl::make_unexpected( "Cannot parse filename: " + filename );
    auto wString = filename.substr( 1, wEndChar - 1 );
    outParams.dimensions.x = std::atoi( wString.c_str() );

    auto hEndChar = filename.find( "_", wEndChar + 1 );
    if ( hEndChar == std::string::npos )
        return tl::make_unexpected( "Cannot parse filename: " + filename );
    auto hString = filename.substr( wEndChar + 2, hEndChar - ( wEndChar + 2 ) );
    outParams.dimensions.y = std::atoi( hString.c_str() );

    auto sEndChar = filename.find( "_", hEndChar + 1 );
    if ( sEndChar == std::string::npos )
        return tl::make_unexpected( "Cannot parse filename: " + filename );
    auto sString = filename.substr( hEndChar + 2, sEndChar - ( hEndChar + 2 ) );
    outParams.dimensions.z = std::atoi( sString.c_str() );

    auto xvEndChar = filename.find( "_", sEndChar + 1 );
    if ( xvEndChar == std::string::npos )
        return tl::make_unexpected( "Cannot parse filename: " + filename );
    auto xvString = filename.substr( sEndChar + 2, xvEndChar - ( sEndChar + 2 ) );
    outParams.voxelSize.x = float(std::atof( xvString.c_str() ) / 1000); // convert mm to meters

    if ( filename[xvEndChar + 1] == 'G' )
    {
        auto gtEndChar = filename.find( "_", xvEndChar + 1 );
        if ( gtEndChar != std::string::npos )
        {
            auto gtString = filename.substr( xvEndChar + 2, gtEndChar - ( xvEndChar + 2 ) );
            outParams.gridLevelSet = gtString == "1"; // convert mm to meters
        }
    }
    if ( filename[xvEndChar + 1] == 'F' ) // end of prefix
    {
        outParams.voxelSize.y = outParams.voxelSize.z = outParams.voxelSize.x;
    }
    else
    {
        auto yvEndChar = filename.find( "_", xvEndChar + 1 );
        if ( yvEndChar == std::string::npos )
            return tl::make_unexpected( "Cannot parse filename: " + filename );
        auto yvString = filename.substr( xvEndChar + 1, yvEndChar - ( xvEndChar + 1 ) );
        outParams.voxelSize.y = float( std::atof( yvString.c_str() ) / 1000 ); // convert mm to meters

        auto zvEndChar = filename.find( "_", yvEndChar + 1 );
        if ( zvEndChar == std::string::npos )
            return tl::make_unexpected( "Cannot parse filename: " + filename );
        auto zvString = filename.substr( yvEndChar + 1, zvEndChar - ( yvEndChar + 1 ) );
        outParams.voxelSize.z = float( std::atof( zvString.c_str() ) / 1000 ); // convert mm to meters

        auto gtEndChar = filename.find( "_", zvEndChar + 1 );
        if ( gtEndChar != std::string::npos )
        {
            auto gtString = filename.substr( zvEndChar + 2, gtEndChar - ( zvEndChar + 2 ) );
            outParams.gridLevelSet = gtString == "1"; // convert mm to meters
        }
    }
    outParams.scalarType = RawParameters::ScalarType::Float32;

    return fromRaw( filepathToOpen, outParams, cb );
}

tl::expected<std::vector<VdbVolume>, std::string> fromVdb( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
{
    if ( cb && !cb( 0.f ) )
        return tl::make_unexpected( getCancelMessage( path ) );
    openvdb::io::File file( path.string() );
    openvdb::initialize();
    file.open();
    std::vector<VdbVolume> res;
    auto grids = file.getGrids();
    file.close();
    if ( grids )
    {
        auto& gridsRef = *grids;
        if ( grids->size() == 0 )
            tl::make_unexpected( std::string( "Nothing to load" ) );

        bool anyLoaded = false;
        int size = int( gridsRef.size() );
        int i = 0;
        ProgressCallback scaledCb;
        if ( cb )
            scaledCb = [cb, &i, size] ( float v ) { return cb( ( i + v ) / size ); };
        for ( i = 0; i < size; ++i )
        {
            if ( !gridsRef[i] )
                continue;

            OpenVdbFloatGrid ovfg( std::move( *std::dynamic_pointer_cast< openvdb::FloatGrid >( gridsRef[i] ) ) );
            VdbVolume vdbVolume;
            vdbVolume.data = std::make_shared<OpenVdbFloatGrid>( std::move( ovfg ) );

            if ( !vdbVolume.data )
                continue;

            const auto dims = vdbVolume.data->evalActiveVoxelDim();
            const auto voxelSize = vdbVolume.data->voxelSize();
            for ( int j = 0; j < 3; ++j )
            {
                vdbVolume.dims[j] = dims[j];
                vdbVolume.voxelSize[j] = float( voxelSize[j] );
            }
            evalGridMinMax( vdbVolume.data, vdbVolume.min, vdbVolume.max );

            if ( scaledCb && !scaledCb( 0.1f ) )
                return tl::make_unexpected( getCancelMessage( path ) );

            openvdb::math::Transform::Ptr transformPtr = std::make_shared<openvdb::math::Transform>();
            vdbVolume.data->setTransform( transformPtr );

            translateToZero( *vdbVolume.data );

            if ( cb && !cb( (1.f + i ) / size ) )
                return tl::make_unexpected( getCancelMessage( path ) );

            res.emplace_back( std::move( vdbVolume ) );

            anyLoaded = true;
        }
        if ( !anyLoaded )
            tl::make_unexpected( std::string( "No loaded grids" ) );
    }
    else
        tl::make_unexpected( std::string( "Nothing to read" ) );

    if ( cb )
        cb( 1.f );

    return res;
}

inline tl::expected<std::vector<VdbVolume>, std::string> toSingleElementVector( tl::expected<VdbVolume, std::string> v )
{
    if ( !v.has_value() )
        return tl::make_unexpected( std::move( v.error() ) );
    return std::vector<VdbVolume>{ std::move( v.value() ) };
}

tl::expected<std::vector<VdbVolume>, std::string> fromAnySupportedFormat( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
{
    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext == ".raw" )
        return toSingleElementVector( fromRaw( path, cb ) );
    if ( ext == ".vdb" )
        return fromVdb( path, cb );
    if ( ext == ".gav" )
        return toSingleElementVector( fromGav( path, cb ) );

    return tl::make_unexpected( std::string( "Unsupported file extension" ) );
}

struct TiffParams
{
    int bitsPerSample = 0;
    int samplesPerPixel = 0;
    int width = 0;
    int height = 0;

    bool operator==( const TiffParams& other ) const
    {
        return bitsPerSample == other.bitsPerSample &&
            samplesPerPixel == other.samplesPerPixel &&
            width == other.width &&
            height == other.height;
    }

    bool operator !=( const TiffParams& other ) const
    {
        return !( *this == other );
    }
};

#ifndef MRMESH_NO_TIFF
// if headerOnly is true, only check that file is valid tiff
// otherwise load whole file
std::tuple<TIFF*, TiffParams> OpenTiff( const std::filesystem::path& path, bool headerOnly = false )
{
    TiffParams tp;
    TIFF* tif = TIFFOpen( MR::utf8string( path ).c_str(), headerOnly ? "rh" : "r" );
    if ( !tif )
        return { tif, tp };

    TIFFGetField( tif, TIFFTAG_BITSPERSAMPLE, &tp.bitsPerSample );
    TIFFGetField( tif, TIFFTAG_SAMPLESPERPIXEL, &tp.samplesPerPixel );
    TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &tp.height );
    if ( !headerOnly )
        tp.width = int( TIFFScanlineSize( tif ) / ( tp.samplesPerPixel * ( tp.bitsPerSample >> 3 ) ) );

    return { tif, tp };
}

bool isTIFFFile( const std::filesystem::path& path )
{
    auto [tif, tp] = OpenTiff( path, true );
    if ( !tif )
        return false;
    TIFFClose( tif );
    return true;
}

template<typename SampleType>
bool ReadVoxels( SimpleVolume& outVolume, size_t layerIndex, TIFF* tif, const TiffParams& tp, float& min, float& max )
{
    assert( sizeof( SampleType ) == tp.bitsPerSample >> 3 );

    std::vector<SampleType> scanline ( tp.width * tp.samplesPerPixel );
    float* pData = &outVolume.data[layerIndex * tp.width * tp.height];

    for ( uint32_t i = 0; i < uint32_t(tp.height); ++i )
    {
        TIFFReadScanline( tif, ( void* )( &scanline[0] ), i );
        for ( size_t j = 0; j < tp.width; ++j )
        {
            float voxel = 0;

            switch ( tp.samplesPerPixel )
            {
            case 1:
                voxel = float( scanline[j] );
                break;
            case 3:
            case 4:            
                voxel =  
                    ( 0.299f * scanline[tp.samplesPerPixel * j] +
                    0.587f * scanline[tp.samplesPerPixel * j + 1] +
                    0.114f * scanline[tp.samplesPerPixel * j + 2] );
                break;
            
            default:
                return false;
            }

            if ( voxel < min )
                min = voxel;

            if ( voxel > max )
                max = voxel;

            pData[j] = voxel;
        }

        pData += tp.width;
    }
    
    return true;
}

tl::expected<VdbVolume, std::string> loadTiffDir( const LoadingTiffSettings& settings )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( settings.dir, ec ) )
        return tl::make_unexpected( "Given path is not directory" );

    int filesNum = 0;
    std::vector<std::filesystem::path> files;
    const std::filesystem::directory_iterator dirEnd;
    for ( auto it = std::filesystem::directory_iterator( settings.dir, ec ); !ec && it != dirEnd; it.increment( ec ) )
    {
        if ( it->is_regular_file( ec ) )
            ++filesNum;
    }
    files.reserve( filesNum );
    for ( auto it = std::filesystem::directory_iterator( settings.dir, ec ); !ec && it != dirEnd; it.increment( ec ) )
    {
        auto filePath = it->path();
        if ( it->is_regular_file( ec ) && isTIFFFile( filePath ) )
            files.push_back( filePath );
    }

    if ( files.size() < 2 )
        return tl::make_unexpected( "Too few TIFF files in the directory" );
    
    sortFilesByName( files );

    auto [tif, tp] = OpenTiff( files.front() );

    SimpleVolume outVolume;
    outVolume.dims = { tp.width, tp.height, int( files.size() ) };
    outVolume.min = FLT_MAX;
    outVolume.max = FLT_MIN;

    outVolume.voxelSize = settings.voxelSize;
    outVolume.data.resize( outVolume.dims.x * outVolume.dims.y * outVolume.dims.z );
    
    for ( int layerIndex = 0; layerIndex < files.size(); ++layerIndex )
    {
        if ( settings.cb && !settings.cb( float( layerIndex ) / files.size() ) )
            return tl::make_unexpected( "Loading was cancelled" );

        switch ( tp.bitsPerSample )
        {
        case 8:
            if ( !ReadVoxels<uint8_t>( outVolume, layerIndex, tif, tp, outVolume.min, outVolume.max ) )
                return tl::make_unexpected( "Unsupported pixel format " );
            break;
        case 16:
            if ( !ReadVoxels<uint16_t>( outVolume, layerIndex, tif, tp, outVolume.min, outVolume.max ) )
                return tl::make_unexpected( "Unsupported pixel format " );
            break;
        default:
            return tl::make_unexpected( "Unsupported pixel format " );
        }
        
        TIFFClose( tif );
        tif = nullptr;

        if ( layerIndex + 1 < files.size() )
        {
            TiffParams currentTiffParams;
            std::tie( tif, currentTiffParams ) = OpenTiff( files[layerIndex + 1] );
            if ( currentTiffParams != tp )
                return tl::make_unexpected( "Unable to process images with different params" );
        }
    }
    
    if ( settings.cb && !settings.cb( 1.0f ) )
        return tl::make_unexpected( "Loading was cancelled" );

    if ( outVolume.data.empty() )
        return tl::make_unexpected( "No voxel data" );

    VdbVolume res;

    res.data = simpleVolumeToDenseGrid( outVolume );
    res.dims = outVolume.dims;
    res.voxelSize = outVolume.voxelSize;
    res.min = outVolume.min;
    res.max = outVolume.max;

    if ( settings.gridType == GridType::LevelSet )
    {
        res.data->setGridClass( openvdb::GridClass::GRID_LEVEL_SET );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, res.dims.x, res.dims.y, 0 ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, res.dims.x, 0, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, 0, res.dims.y, res.dims.z ), res.max, false );

        res.data->denseFill( openvdb::CoordBBox( 0, 0, res.dims.z, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, res.dims.y, 0, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( res.dims.x, 0, 0, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
    }
    
    return res;
}
#endif // MRMESH_NO_TIFF

tl::expected<VdbVolume, std::string> fromRaw( const std::filesystem::path& file, const RawParameters& params,
    const ProgressCallback& cb )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
    return addFileNameInError( fromRaw( in, params, cb ), file );
}

tl::expected<VdbVolume, std::string> fromRaw( std::istream& in, const RawParameters& params,  const ProgressCallback& cb )
{
    if ( params.dimensions.x <= 0 || params.dimensions.y <= 0 || params.dimensions.z <= 0 )
        return tl::make_unexpected( "Wrong volume dimension parameter value" );

    if ( params.voxelSize.x <= 0 || params.voxelSize.y <= 0 || params.voxelSize.z <= 0 )
        return tl::make_unexpected( "Wrong voxel size parameter value" );

    int unitSize = 0;
    switch ( params.scalarType )
    {
    case RawParameters::ScalarType::UInt8:
        unitSize = 1;
        break;
    case RawParameters::ScalarType::Int8:
        unitSize = 1;
        break;
    case RawParameters::ScalarType::UInt16:
        unitSize = 2;
        break;
    case RawParameters::ScalarType::Int16:
        unitSize = 2;
        break;
    case RawParameters::ScalarType::UInt32:
        unitSize = 4;
        break;
    case RawParameters::ScalarType::Int32:
        unitSize = 4;
        break;
    case RawParameters::ScalarType::Float32:
        unitSize = 4;
    break; 
    case RawParameters::ScalarType::UInt64:
        unitSize = 8;
        break;
    case RawParameters::ScalarType::Int64:
        unitSize = 8;
        break;
    case RawParameters::ScalarType::Float64:
        unitSize = 8;
        break;
    default:
        assert( false );
        return tl::make_unexpected( "Wrong scalar type parameter value" );
    }

    SimpleVolume outVolume;
    outVolume.dims = params.dimensions;
    outVolume.voxelSize = params.voxelSize;
    outVolume.data.resize( size_t( outVolume.dims.x ) * outVolume.dims.y * outVolume.dims.z );
    char* outPointer{ nullptr };
    std::vector<char> data;
    if ( params.scalarType == RawParameters::ScalarType::Float32 )
        outPointer = ( char* )outVolume.data.data();
    else
    {
        data.resize( outVolume.data.size() * unitSize );
        outPointer = data.data();
    }

    size_t xyDimsUnit = size_t( params.dimensions.x ) * params.dimensions.y * unitSize;
    for ( int z = 0; z < params.dimensions.z; ++z )
    {
        size_t shift = xyDimsUnit * z;
        if ( !in.read( outPointer + shift, xyDimsUnit ) )
            return tl::make_unexpected( "Read error" );
        if ( cb )
            cb( ( z + 1.0f ) / float( params.dimensions.z ) );
    }

    if ( params.scalarType != RawParameters::ScalarType::Float32 )
    {
        int64_t min = 0;
        uint64_t max = 0;
        if ( params.scalarType == RawParameters::ScalarType::Int8 )
        {
            min = std::numeric_limits<int8_t>::lowest();
            max = std::numeric_limits<int8_t>::max();
        }
        else if ( params.scalarType == RawParameters::ScalarType::Int16 )
        {
            min = std::numeric_limits<int16_t>::lowest();
            max = std::numeric_limits<int16_t>::max();
        }
        else if ( params.scalarType == RawParameters::ScalarType::Int32 )
        {
            min = std::numeric_limits<int32_t>::lowest();
            max = std::numeric_limits<int32_t>::max();
        }
        else if ( params.scalarType == RawParameters::ScalarType::Int64 )
        {
            min = std::numeric_limits<int64_t>::lowest();
            max = std::numeric_limits<int64_t>::max();
        }
        else if ( params.scalarType == RawParameters::ScalarType::UInt8 )
            max = std::numeric_limits<uint8_t>::max();
        else if ( params.scalarType == RawParameters::ScalarType::UInt16 )
            max = std::numeric_limits<uint16_t>::max();
        else if ( params.scalarType == RawParameters::ScalarType::UInt32 )
            max = std::numeric_limits<uint32_t>::max();
        else if ( params.scalarType == RawParameters::ScalarType::UInt64 )
            max = std::numeric_limits<uint64_t>::max();
        auto converter = getTypeConverter( params.scalarType, max - min, min );
        for ( size_t i = 0; i < outVolume.data.size(); ++i )
        {
            float value = converter( &outPointer[i * unitSize] );
            outVolume.data[i] = value;
            outVolume.max = std::max( outVolume.max, value );
            outVolume.min = std::min( outVolume.min, value );
        }
    }
    else
    {
        auto minmaxIt = std::minmax_element( outVolume.data.begin(), outVolume.data.end() );
        outVolume.min = *minmaxIt.first;
        outVolume.max = *minmaxIt.second;
    }

    VdbVolume res;
    res.data = simpleVolumeToDenseGrid( outVolume );
    if ( params.gridLevelSet )
    {
        openvdb::tools::changeBackground( res.data->tree(), outVolume.max );
        res.data->setGridClass( openvdb::GRID_LEVEL_SET );
    }
    res.dims = outVolume.dims;
    res.voxelSize = outVolume.voxelSize;
    res.min = outVolume.min;
    res.max = outVolume.max;
    return res;
}

}
}
#endif // !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXELS )
