#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_DICOM )
#include "MRVoxelsLoad.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRObjectVoxels.h"
#include "MRVDBConversions.h"
#include "MRStringConvert.h"
#include "MRFloatGrid.h"
#include <gdcmImageHelper.h>
#include <gdcmImageReader.h>
#include <gdcmTagKeywords.h>
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <compare>
#include <filesystem>

namespace MR
{

namespace VoxelsLoad
{

const IOFilters Filters =
{
    {"Raw (.raw)","*.raw"}
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

std::function<float( char* )> getTypeConverter( const gdcm::PixelFormat& format, const uint64_t& range, const int64_t& min )
{
    switch ( gdcm::PixelFormat::ScalarType( format ) )
    {
    case gdcm::PixelFormat::UINT8:
        return [range, min]( char* c )
        {
            return float( *(uint8_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::UINT16:
        return [range, min]( char* c )
        {
            return float( *(uint16_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::INT8:
        return [range, min]( char* c )
        {
            return float( *(int8_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::INT16:
        return [range, min]( char* c )
        {
            return float( *(int16_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::INT32:
        return [range, min]( char* c )
        {
            return float( *(int32_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::UINT32:
        return [range, min]( char* c )
        {
            return float( *(uint32_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::UINT64:
        return [range, min]( char* c )
        {
            return float( *(uint64_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::INT64:
        return [range, min]( char* c )
        {
            return float( *(int64_t*) (c) -min ) / float( range );
        };
    case gdcm::PixelFormat::FLOAT32:
        return []( char* c )
        {
            return *(float*) ( c );
        };
    case gdcm::PixelFormat::FLOAT64:
        return []( char* c )
        {
            return float( *(double*) ( c ) );
        };
    default:
        break;
    }
    return {};
}

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
        gdcm::Tag( 0x0028, 0x0004 ),// is for PhotometricInterpretation
        gdcm::Keywords::ImagePositionPatient::GetTag(), // is for image origin in mm
        gdcm::Tag( 0x0028, 0x0010 ),gdcm::Tag( 0x0028, 0x0011 ),gdcm::Tag( 0x0028, 0x0008 )}; // is for dimensions
    if ( !ir.ReadSelectedTags( tags ) )
        return false;
    gdcm::MediaStorage ms;
    //spdlog::info( "File {}: ms={}", utf8string( path ), (int)ms );
    ms.SetFromFile( ir.GetFile() );
    if ( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage || ms == gdcm::MediaStorage::SecondaryCaptureImageStorage )
        return false;
    auto photometric = gdcm::ImageHelper::GetPhotometricInterpretationValue( ir.GetFile() );
    if ( photometric != gdcm::PhotometricInterpretation::MONOCHROME2 &&
         photometric != gdcm::PhotometricInterpretation::MONOCHROME1 )
        return false;
    auto dims = gdcm::ImageHelper::GetDimensionsValue( ir.GetFile() );
    assert( dims.size() == 3 );
    return true;
}

struct DCMFileLoadResult
{
    bool success = false;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    std::string seriesDescription;
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
        spdlog::error( "loadSingle: cannot read file: {}", utf8string( path ) );
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
    auto caster = getTypeConverter( gimage.GetPixelFormat(), max - min, min );
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

tl::expected<LoadDCMResult, std::string> loadDCMFolder( const std::filesystem::path& path,
                                                    unsigned maxNumThreads, ProgressCallback cb )
{
    MR_TIMER;
    ProgressCallback newCb{};
    if ( cb )
        newCb = [&cb](float f) { return cb( 0.5f * f ); };
    if ( newCb && !newCb( 0.0f ) )
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
        if ( newCb && !newCb( 0.3f * float( fCounter ) / float( filesNum ) ) )
            tl::make_unexpected( "Loading canceled" );
    }
    if ( files.empty() )
    {
        return tl::make_unexpected( "loadDCMFolder: there is no dcm file in folder: " + utf8string( path ) );
    }
    if ( files.size() == 1 )
    {
        ProgressCallback localCb{};
        if ( newCb )
            localCb = [&newCb]( float f ) { return newCb( 0.3f + 0.7f * f ); };
        return loadDCMFile( files[0], localCb );
    }
    sortDICOMFiles( files, maxNumThreads, data.voxelSize );
    data.dims.z = (int) files.size();

    auto firstRes = loadSingleFile( files.front(), data, 0 );
    if ( !firstRes.success )
        return tl::make_unexpected( "loadDCMFolder: error" );
    data.min = firstRes.min;
    data.max = firstRes.max;
    size_t dimXY = data.dims.x * data.dims.y;

    if ( newCb && !newCb( 0.4f ) )
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
                if ( newCb && std::this_thread::get_id() == mainThreadId )
                    cancelCalled = !newCb( 0.4f + 0.6f * ( float( numLoadedSlices ) / float( slicesRes.size() ) ) );
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

    if ( cb )
        newCb = [&cb] ( float f ) { return cb( 0.5f + 0.5f * f ); };
    LoadDCMResult res;
    res.vdbVolume = simpleVolumeToVdbVolume( data, newCb );
    if ( firstRes.seriesDescription.empty() )
        res.name = utf8string( files.front().stem() );
    else
        res.name = firstRes.seriesDescription;
    return res;
}

std::vector<tl::expected<LoadDCMResult, std::string>> loadDCMFolderTree( const std::filesystem::path& path, unsigned maxNumThreads, ProgressCallback cb )
{
    MR_TIMER;
    std::vector<tl::expected<LoadDCMResult, std::string>> res;
    bool anySuccessLoad{ false };
    auto tryLoadDir = [&]( const std::filesystem::path& dir )
    {
        auto loadRes = loadDCMFolder( dir, maxNumThreads, cb );
        if ( loadRes.has_value() )
        {
            res.push_back( *loadRes );
            anySuccessLoad = true;
        }
        else
        {
            const std::string str = "loadDCMFolder: there is no dcm file in folder:";
            if ( anySuccessLoad && loadRes.error().substr( 0, str.size() ) == str )
                return true;
            res.push_back( loadRes );
            if ( loadRes.error() == "Loading canceled" )
                return false;
        }
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

tl::expected<LoadDCMResult, std::string> loadDCMFile( const std::filesystem::path& path, ProgressCallback cb )
{
    MR_TIMER;
    ProgressCallback newCb{};
    if ( cb )
        newCb = [&cb]( float f ) { return 0.5f * cb( f ); };
    if ( newCb && !newCb( 0.0f ) )
        return tl::make_unexpected( "Loading canceled" );

    SimpleVolume simpleVolume;
    simpleVolume.voxelSize = Vector3f();
    simpleVolume.dims.z = 1;
    auto fileRes = loadSingleFile( path, simpleVolume, 0 );
    if ( !fileRes.success )
        return tl::make_unexpected( "loadDCMFile: error load file: " + utf8string( path ) );
    if ( newCb && !newCb( 0.5f ) )
        return tl::make_unexpected( "Loading canceled" );
    simpleVolume.max = fileRes.max;
    simpleVolume.min = fileRes.min;
    
    if ( cb )
        newCb = [&cb] ( float f ) { return cb( 0.5f + 0.5f * f ); };
    LoadDCMResult res;
    res.vdbVolume = simpleVolumeToVdbVolume( simpleVolume, newCb );
    res.name = utf8string( path.stem() );
    return res;
}

tl::expected<VdbVolume, std::string> loadRaw( const std::filesystem::path& path,
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
            auto gtString = filename.substr( zvEndChar + 1, gtEndChar - ( zvEndChar + 1 ) );
            outParams.gridLevelSet = gtString == "1"; // convert mm to meters
        }
    }
    outParams.scalarType = RawParameters::ScalarType::Float32;

    return loadRaw( filepathToOpen, outParams, cb );
}

tl::expected<VdbVolume, std::string> loadRaw( const std::filesystem::path& path, const RawParameters& params,
    const ProgressCallback& cb )
{
    if ( params.dimensions.x <= 0 || params.dimensions.y <= 0 || params.dimensions.z <= 0 ||
        params.voxelSize.x == 0.0f || params.voxelSize.y == 0.0f || params.voxelSize.z == 0.0f )
        return tl::make_unexpected( "Bad parameters for reading " + utf8string( path.filename() ) );
    SimpleVolume outVolume;
    outVolume.dims = params.dimensions;
    outVolume.voxelSize = params.voxelSize;

    int unitSize = 0;
    gdcm::PixelFormat format = gdcm::PixelFormat::FLOAT32;
    switch ( params.scalarType )
    {
    case RawParameters::ScalarType::UInt8:
        format = gdcm::PixelFormat::UINT8;
        unitSize = 1;
        break;
    case RawParameters::ScalarType::Int8:
        format = gdcm::PixelFormat::INT8;
        unitSize = 1;
        break;
    case RawParameters::ScalarType::UInt16:
        format = gdcm::PixelFormat::UINT16;
        unitSize = 2;
        break;
    case RawParameters::ScalarType::Int16:
        format = gdcm::PixelFormat::INT16;
        unitSize = 2;
        break;
    case RawParameters::ScalarType::UInt32:
        format = gdcm::PixelFormat::UINT32;
        unitSize = 4;
        break;
    case RawParameters::ScalarType::Int32:
        format = gdcm::PixelFormat::INT32;
        unitSize = 4;
        break;
    case RawParameters::ScalarType::Float32:
        format = gdcm::PixelFormat::FLOAT32;
        unitSize = 4;
    break; 
    case RawParameters::ScalarType::UInt64:
        format = gdcm::PixelFormat::UINT64;
        unitSize = 8;
        break;
    case RawParameters::ScalarType::Int64:
        format = gdcm::PixelFormat::INT64;
        unitSize = 8;
        break;
    case RawParameters::ScalarType::Float64:
        format = gdcm::PixelFormat::FLOAT64;
        unitSize = 8;
        break;
    default:
        assert( false );
        return tl::make_unexpected( "Bad parameters for reading " + utf8string( path.filename() ) );
    }

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
    std::ifstream infile( path, std::ios::binary );
    int xyDimsUnit = params.dimensions.x * params.dimensions.y * unitSize;
    for ( int z = 0; z < params.dimensions.z; ++z )
    {
        int shift = xyDimsUnit * z;
        if ( !infile.read( outPointer + shift, xyDimsUnit ) )
            return tl::make_unexpected( "Cannot read file: " + utf8string( path ) );
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
        auto converter = getTypeConverter( format, max - min, min );
        for ( int i = 0; i < outVolume.data.size(); ++i )
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
        res.data->setGridClass( openvdb::GRID_LEVEL_SET );
    res.dims = outVolume.dims;
    res.voxelSize = outVolume.voxelSize;
    res.min = outVolume.min;
    outVolume.max = outVolume.max;
    return res;
}

}
}
#endif
