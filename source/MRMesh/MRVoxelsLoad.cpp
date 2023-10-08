#include "MRVoxelsLoad.h"
#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRObjectVoxels.h"
#include "MRVDBConversions.h"
#include "MRStringConvert.h"
#include "MRVDBFloatGrid.h"
#include "MRStringConvert.h"
#include "MRDirectory.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <cfloat>
#include <compare>
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

#include "MROpenVDBHelper.h"
#include "MRTiffIO.h"
#include "MRParallelFor.h"

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
bool isDICOMFile( const std::filesystem::path& path, std::string& seriesUid )
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
        gdcm::Keywords::ImagePositionPatient::GetTag(), // is for image origin in mm,
        gdcm::Keywords::SeriesInstanceUID::GetTag(),
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

    const gdcm::DataSet& ds = ir.GetFile().GetDataSet();
    if ( ds.FindDataElement( gdcm::Keywords::SeriesInstanceUID::GetTag() ) )
    {
        const gdcm::DataElement& de = ds.GetDataElement( gdcm::Keywords::SeriesInstanceUID::GetTag() );
        gdcm::Keywords::SeriesInstanceUID uid;
        uid.SetFromDataElement( de );
        auto uidVal = uid.GetValue();
        seriesUid = uidVal;
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
        spdlog::error( "PhotometricInterpretation: {}", (int)gimage.GetPhotometricInterpretation() );
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
        spdlog::error( "Type: {}", (int)gimage.GetPixelFormat() );
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

struct SeriesInfo
{
    float sliceSize{ 0.0f };
    int numSlices{ 0 };
    BitSet missedSlices;
};

SeriesInfo sortDICOMFiles( std::vector<std::filesystem::path>& files, unsigned maxNumThreads )
{
    SeriesInfo res;

    if ( files.empty() )
        return res;

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
        auto denom = std::max( 1.0f, float( zOrder[1].instanceNum - zOrder[0].instanceNum ) );
        res.sliceSize = float( ( zOrder[1].imagePos - zOrder[0].imagePos ).length() / denom / 1000.0 );
        res.numSlices = zOrder.back().instanceNum - zOrder.front().instanceNum + 1;

        bool needReverse = zOrder[1].imagePos.z < zOrder[0].imagePos.z;

        if ( res.numSlices != 0 )
        {
            res.missedSlices.resize( res.numSlices );

            auto startIN = zOrder[0].instanceNum;
            for ( int i = 1; i < zOrder.size(); ++i )
            {
                auto prevIN = zOrder[i - 1].instanceNum;
                auto diff = zOrder[i].instanceNum - prevIN;
                if ( diff == 1 )
                    continue;
                if ( diff == 0 )
                {
                    res.numSlices = 0;
                    res.missedSlices.clear();
                    break; // non-consistent instances
                }

                int startMissedIndex = ( prevIN - startIN + 1 );
                for ( int j = startMissedIndex; j + 1 < startMissedIndex + diff; ++j )
                    res.missedSlices.set( needReverse ? ( res.numSlices - 1 - j ) : j );
            }
        }

        // if slices go in descending z-order then reverse them
        if ( needReverse )
            std::reverse( files.begin(), files.end() );
    }
    return res;
}

Expected<DicomVolume, std::string> loadSingleDicomFolder( std::vector<std::filesystem::path>& files,
                                                        unsigned maxNumThreads, const ProgressCallback& cb )
{
    MR_TIMER
    if ( !reportProgress( cb, 0.0f ) )
        return unexpected( "Loading canceled" );

    if ( files.empty() )
        return unexpected( "loadDCMFolder: there is no dcm file" );

    SimpleVolume data;
    data.voxelSize = Vector3f();
    data.dims = Vector3i::diagonal( 0 );

    if ( files.size() == 1 )
        return loadDicomFile( files[0], subprogress( cb, 0.3f, 1.0f ) );
    auto seriesInfo = sortDICOMFiles( files, maxNumThreads );
    if ( seriesInfo.sliceSize != 0.0f )
        data.voxelSize.z = seriesInfo.sliceSize;

    if ( seriesInfo.numSlices == 0 )
        data.dims.z = ( int )files.size();
    else
        data.dims.z = seriesInfo.numSlices;

    auto firstRes = loadSingleFile( files.front(), data, 0 );
    if ( !firstRes.success )
        return unexpected( "loadDCMFolder: error loading first file \"" + utf8string( files.front() ) + "\"" );
    data.min = firstRes.min;
    data.max = firstRes.max;
    size_t dimXY = data.dims.x * data.dims.y;

    if ( !reportProgress( cb, 0.4f ) )
        return unexpected( "Loading canceled" );

    auto presentSlices = seriesInfo.missedSlices; 
    presentSlices.resize( data.dims.z );
    presentSlices.flip();

    // other slices
    bool cancelCalled = false;
    std::vector<DCMFileLoadResult> slicesRes( files.size() - 1 );
    tbb::task_arena limitedArena( maxNumThreads );
    std::atomic<int> numLoadedSlices = 0;
    limitedArena.execute( [&]
    {
        cancelCalled = !ParallelFor( 0, int( slicesRes.size() ), [&] ( int i )
        {
            slicesRes[i] = loadSingleFile( files[i + 1], data, ( presentSlices.nthSetBit( i + 1 ) ) * dimXY );
            ++numLoadedSlices;
        }, subprogress( cb, 0.4f, 0.9f ), 1 );
    } );
    if ( cancelCalled )
        return unexpected( "Loading canceled" );

    // fill missed slices
    int missedSlicesNum = int( seriesInfo.missedSlices.count() );
    if ( missedSlicesNum != 0 )
    {
        int passedSlices = 0;
        int prevPresentSlice = -1;
        for ( auto presentSlice : presentSlices )
        {
            int numMissed = int( presentSlice ) - prevPresentSlice - 1;
            if ( numMissed == 0 )
            {
                prevPresentSlice = int( presentSlice );
                continue;
            }

            auto sb = subprogress( cb,
                0.9f + 0.1f * float( passedSlices ) / float( missedSlicesNum ),
                0.9f + 0.1f * float( passedSlices + numMissed ) / float( missedSlicesNum ) );
            int firstMissed = prevPresentSlice + 1;
            float ratioDenom = 1.0f / float( numMissed + 1 );
            cancelCalled = !ParallelFor( firstMissed * dimXY, presentSlice * dimXY, [&] ( size_t i )
            {
                auto posZ = int( i / dimXY );
                auto zBotDiff = posZ - prevPresentSlice;
                auto botValue = data.data[i - dimXY * zBotDiff];
                auto topValue = data.data[i + dimXY * ( int( presentSlice ) - posZ )];
                float ratio = float( zBotDiff ) * ratioDenom;
                data.data[i] = botValue * ( 1.0f - ratio ) + topValue * ratio;
            }, sb );
            if ( cancelCalled )
                break;
            prevPresentSlice = int( presentSlice );
            passedSlices += numMissed;
        }
    }

    if ( cancelCalled )
        return unexpected( "Loading canceled" );

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

using SeriesMap = std::unordered_map<std::string, std::vector<std::filesystem::path>>;

Expected<SeriesMap,std::string> extractDCMSeries( const std::filesystem::path& path,
    const ProgressCallback& cb )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( path, ec ) )
        return { unexpected( "loadDCMFolder: path is not directory" ) };

    int filesNum = 0;
    std::vector<std::filesystem::path> files;
    for ( auto entry : Directory{ path, ec } )
    {
        if ( entry.is_regular_file( ec ) )
            ++filesNum;
    }

    std::unordered_map<std::string, std::vector<std::filesystem::path>> seriesMap;
    int fCounter = 0;
    for ( auto entry : Directory{ path, ec } )
    {
        ++fCounter;
        auto filePath = entry.path();
        std::string uid;
        if ( entry.is_regular_file( ec ) && isDICOMFile( filePath, uid ) )
            seriesMap[uid].push_back( filePath );
        if ( !reportProgress( cb, float( fCounter ) / float( filesNum ) ) )
            return { unexpected( "Loading canceled" ) };
    }

    if ( seriesMap.empty() )
        return unexpected( "No dcm series in folder: " + utf8string( path ) );

    return seriesMap;
}

std::vector<Expected<DicomVolume, std::string>> loadDicomsFolder( const std::filesystem::path& path,
                                                        unsigned maxNumThreads, const ProgressCallback& cb )
{
    auto seriesMap = extractDCMSeries( path, subprogress( cb, 0.0f, 0.3f ) );
    if ( !seriesMap.has_value() )
        return { unexpected( seriesMap.error() ) };

    int seriesCounter = 0;
    auto seriesNum = seriesMap->size();
    std::vector<Expected<DicomVolume, std::string>> res;
    for ( auto& [uid, series] : *seriesMap )
    {
        res.push_back( loadSingleDicomFolder( series, maxNumThreads,
            subprogress( cb,
                0.3f + 0.7f * float( seriesCounter ) / float( seriesNum ),
                0.3f + 0.7f * float( seriesCounter + 1 ) / float( seriesNum ) ) ) );

        ++seriesCounter;
        if ( !res.back().has_value() && res.back().error() == "Loading canceled" )
            return { unexpected( "Loading canceled" ) };
    }
    return res;
}

Expected<MR::VoxelsLoad::DicomVolume, std::string> loadDicomFolder( const std::filesystem::path& path, unsigned maxNumThreads /*= 4*/, const ProgressCallback& cb /*= {} */ )
{
    auto seriesMap = extractDCMSeries( path, subprogress( cb, 0.0f, 0.3f ) );
    if ( !seriesMap.has_value() )
        return { unexpected( seriesMap.error() ) };

    return loadSingleDicomFolder( seriesMap->begin()->second, maxNumThreads, subprogress( cb, 0.3f, 1.0f ) );
}

std::vector<Expected<LoadDCMResult, std::string>> loadDCMsFolder( const std::filesystem::path& path,
                                                     unsigned maxNumThreads, const ProgressCallback& cb )
{
    auto dicomRes = loadDicomsFolder( path, maxNumThreads, subprogress( cb, 0, 0.5f ) );
    std::vector<Expected<LoadDCMResult, std::string>> res( dicomRes.size() );
    for ( int i = 0; i < dicomRes.size(); ++i )
    {
        if ( !dicomRes[i].has_value() )
        {
            res[i] = unexpected( std::move( dicomRes[i].error() ) );
            continue;
        }
        res[i] = LoadDCMResult();
        res[i]->vdbVolume = simpleVolumeToVdbVolume( std::move( dicomRes[i]->vol ),
            subprogress( cb, 
                0.5f + float( i ) / float( dicomRes.size() ) * 0.5f, 
                0.5f + float( i + 1 ) / float( dicomRes.size() ) * 0.5f ) );
        res[i]->name = std::move( dicomRes[i]->name );
        res[i]->xf = std::move( dicomRes[i]->xf );
        if ( cb && !cb( 0.5f + float( i + 1 ) / float( dicomRes.size() ) * 0.5f ) )
            return { unexpected( "Loading canceled" ) };
    }

    return { res };
}

Expected<MR::VoxelsLoad::LoadDCMResult, std::string> loadDCMFolder( const std::filesystem::path& path, unsigned maxNumThreads /*= 4*/, const ProgressCallback& cb /*= {} */ )
{
    auto loadRes = loadDicomFolder( path, maxNumThreads, subprogress( cb, 0.0f, 0.5f ) );
    if ( !loadRes.has_value() )
        return unexpected( loadRes.error() );
    LoadDCMResult res;
    res.vdbVolume = simpleVolumeToVdbVolume( std::move( loadRes->vol ), subprogress( cb, 0.5f, 1.0f ) );
    res.name = std::move( loadRes->name );
    res.xf = std::move( loadRes->xf );
    return res;
}

std::vector<Expected<LoadDCMResult, std::string>> loadDCMFolderTree( const std::filesystem::path& path, unsigned maxNumThreads, const ProgressCallback& cb )
{
    MR_TIMER;
    std::vector<Expected<LoadDCMResult, std::string>> res;
    auto tryLoadDir = [&]( const std::filesystem::path& dir )
    {
        auto loadRes = loadDCMsFolder( dir, maxNumThreads, cb );
        if ( loadRes.size() == 1 && !loadRes[0].has_value() && loadRes[0].error() == "Loading canceled" )
            return false;

        res.insert( res.end(), std::make_move_iterator( loadRes.begin() ), std::make_move_iterator( loadRes.end() ) );

        return true;
    };
    if ( !tryLoadDir( path ) )
        return { unexpected( "Loading canceled" ) };

    std::error_code ec;
    for ( auto entry : DirectoryRecursive{ path, ec } )
    {
        if ( entry.is_directory( ec ) && !tryLoadDir( entry ) )
            break;
    }
    return res;
}

Expected<DicomVolume, std::string> loadDicomFile( const std::filesystem::path& path, const ProgressCallback& cb )
{
    MR_TIMER
    if ( !reportProgress( cb, 0.0f ) )
        return unexpected( "Loading canceled" );

    SimpleVolume simpleVolume;
    simpleVolume.voxelSize = Vector3f();
    simpleVolume.dims.z = 1;
    auto fileRes = loadSingleFile( path, simpleVolume, 0 );
    if ( !fileRes.success )
        return unexpected( "loadDCMFile: error load file: " + utf8string( path ) );
    simpleVolume.max = fileRes.max;
    simpleVolume.min = fileRes.min;
    
    DicomVolume res;
    res.vol = std::move( simpleVolume );
    res.name = utf8string( path.stem() );
    return res;
}

#endif // MRMESH_NO_DICOM

Expected<VdbVolume, std::string> fromRaw( const std::filesystem::path& path,
    const ProgressCallback& cb )
{
    MR_TIMER;

    if ( path.empty() )
    {
        return unexpected( "Path is empty" );
    }

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return unexpected( ss.str() );
    }

    auto parentPath = path.parent_path();
    std::error_code ec;
    if ( !std::filesystem::is_directory( parentPath, ec ) )
        return unexpected( utf8string( parentPath ) + " - is not directory" );
    std::vector<std::filesystem::path> candidatePaths;
    for ( auto entry : Directory{ parentPath, ec } )
    {
        auto filename = entry.path().filename();
        auto pos = utf8string( filename ).find( utf8string( path.filename() ) );
        if ( pos != std::string::npos )
            candidatePaths.push_back( entry.path() );
    }
    if ( candidatePaths.empty() )
        return unexpected( "Cannot find file: " + utf8string( path.filename() ) );
    else if ( candidatePaths.size() > 1 )
        return unexpected( "More than one file exists: " + utf8string( path.filename() ) );

    RawParameters outParams;
    auto filepathToOpen = candidatePaths[0];
    auto filename = utf8string( filepathToOpen.filename() );
    auto wEndChar = filename.find("_");
    if ( wEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto wString = filename.substr( 1, wEndChar - 1 );
    outParams.dimensions.x = std::atoi( wString.c_str() );

    auto hEndChar = filename.find( "_", wEndChar + 1 );
    if ( hEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto hString = filename.substr( wEndChar + 2, hEndChar - ( wEndChar + 2 ) );
    outParams.dimensions.y = std::atoi( hString.c_str() );

    auto sEndChar = filename.find( "_", hEndChar + 1 );
    if ( sEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto sString = filename.substr( hEndChar + 2, sEndChar - ( hEndChar + 2 ) );
    outParams.dimensions.z = std::atoi( sString.c_str() );

    auto xvEndChar = filename.find( "_", sEndChar + 1 );
    if ( xvEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
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
            return unexpected( "Cannot parse filename: " + filename );
        auto yvString = filename.substr( xvEndChar + 1, yvEndChar - ( xvEndChar + 1 ) );
        outParams.voxelSize.y = float( std::atof( yvString.c_str() ) / 1000 ); // convert mm to meters

        auto zvEndChar = filename.find( "_", yvEndChar + 1 );
        if ( zvEndChar == std::string::npos )
            return unexpected( "Cannot parse filename: " + filename );
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

Expected<std::vector<VdbVolume>, std::string> fromVdb( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
{
    if ( cb && !cb( 0.f ) )
        return unexpected( getCancelMessage( path ) );
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
            unexpected( std::string( "Nothing to load" ) );

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
                return unexpected( getCancelMessage( path ) );

            openvdb::math::Transform::Ptr transformPtr = std::make_shared<openvdb::math::Transform>();
            vdbVolume.data->setTransform( transformPtr );

            translateToZero( *vdbVolume.data );

            if ( cb && !cb( (1.f + i ) / size ) )
                return unexpected( getCancelMessage( path ) );

            res.emplace_back( std::move( vdbVolume ) );

            anyLoaded = true;
        }
        if ( !anyLoaded )
            unexpected( std::string( "No loaded grids" ) );
    }
    else
        unexpected( std::string( "Nothing to read" ) );

    if ( cb )
        cb( 1.f );

    return res;
}

inline Expected<std::vector<VdbVolume>, std::string> toSingleElementVector( Expected<VdbVolume, std::string> v )
{
    if ( !v.has_value() )
        return unexpected( std::move( v.error() ) );
    return std::vector<VdbVolume>{ std::move( v.value() ) };
}

Expected<std::vector<VdbVolume>, std::string> fromAnySupportedFormat( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
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

    return unexpected( std::string( "Unsupported file extension" ) );
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
Expected<VdbVolume, std::string> loadTiffDir( const LoadingTiffSettings& settings )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( settings.dir, ec ) )
        return unexpected( "Given path is not directory" );

    int filesNum = 0;
    std::vector<std::filesystem::path> files;
    for ( auto entry : Directory{ settings.dir, ec } )
    {
        if ( entry.is_regular_file( ec ) )
            ++filesNum;
    }
    files.reserve( filesNum );
    for ( auto entry : Directory{ settings.dir, ec } )
    {
        auto filePath = entry.path();
        if ( entry.is_regular_file( ec ) && isTIFFFile( filePath ) )
            files.push_back( filePath );
    }

    if ( files.size() < 2 )
        return unexpected( "Too few TIFF files in the directory" );
    
    sortFilesByName( files );

    auto tpExp = readTiffParameters( files.front() );
    if ( !tpExp.has_value() )
        return unexpected( tpExp.error() );

    auto& tp = *tpExp;

    SimpleVolume outVolume;
    outVolume.dims = { tp.imageSize.x, tp.imageSize.y, int( files.size() ) };
    outVolume.min = FLT_MAX;
    outVolume.max = FLT_MIN;

    outVolume.voxelSize = settings.voxelSize;
    outVolume.data.resize( outVolume.dims.x * outVolume.dims.y * outVolume.dims.z );

    TiffParameters localParams;
    RawTiffOutput output;
    output.size = ( tp.imageSize.x * tp.imageSize.y ) * sizeof( float );
    output.params = &localParams;
    output.min = &outVolume.min;
    output.max = &outVolume.max;
    for ( int layerIndex = 0; layerIndex < files.size(); ++layerIndex )
    {
        output.bytes = ( uint8_t* )( outVolume.data.data() + layerIndex * tp.imageSize.x * tp.imageSize.y );
        auto readRes = readRawTiff( files[layerIndex], output );

        if ( !readRes.has_value() )
            return unexpected( readRes.error() );

        if ( localParams != tp )
            return unexpected( "Inconsistent TIFF files" );

        if ( settings.cb && !settings.cb( float( layerIndex ) / files.size() ) )
            return unexpected( "Loading was cancelled" );
    }
    
    if ( settings.cb && !settings.cb( 1.0f ) )
        return unexpected( "Loading was cancelled" );

    if ( outVolume.data.empty() )
        return unexpected( "No voxel data" );

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

Expected<VdbVolume, std::string> fromRaw( const std::filesystem::path& file, const RawParameters& params,
    const ProgressCallback& cb )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
    return addFileNameInError( fromRaw( in, params, cb ), file );
}

Expected<VdbVolume, std::string> fromRaw( std::istream& in, const RawParameters& params,  const ProgressCallback& cb )
{
    if ( params.dimensions.x <= 0 || params.dimensions.y <= 0 || params.dimensions.z <= 0 )
        return unexpected( "Wrong volume dimension parameter value" );

    if ( params.voxelSize.x <= 0 || params.voxelSize.y <= 0 || params.voxelSize.z <= 0 )
        return unexpected( "Wrong voxel size parameter value" );

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
        return unexpected( "Wrong scalar type parameter value" );
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
            return unexpected( "Read error" );
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
