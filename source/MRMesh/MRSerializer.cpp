#include "MRSerializer.h"
#include "MRFile.h"
#include "MRObject.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix3.h"
#include "MRBase64.h"
#include "MRBitSet.h"
#include "MRPlane3.h"
#include "MRTriPoint.h"
#include "MRTimer.h"
#include "MRObjectFactory.h"
#include "MRPointOnFace.h"
#include "MRMeshLoad.h"
#include "MRMeshSave.h"
#include "MRMeshTriPoint.h"
#include "MRMesh.h"
#include "MRStreamOperators.h"
#include "MRCube.h"
#include "MRObjectMesh.h"
#include "MRStringConvert.h"
#include <filesystem>
#include "MRPch/MRSpdlog.h"
#include "MRGTest.h"
#include "MRPch/MRJson.h"

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#endif

#include <zip.h>

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

#include <streambuf>

namespace MR
{


UniqueTemporaryFolder::UniqueTemporaryFolder( FolderCallback onPreTempFolderDelete )
    : onPreTempFolderDelete_( std::move( onPreTempFolderDelete ) )
{
    MR_TIMER;
    std::error_code ec;
    const auto tmp = std::filesystem::temp_directory_path( ec );
    if ( ec )
    {
        spdlog::error( "Cannot get temporary directory: {}", ec.message() );
        return;
    }

    constexpr int MAX_ATTEMPTS = 32;
    // if the process is terminated in between temporary folder creation and removal, then
    // all 32 folders can be present, so we use current time to ignore old folders
    auto t0 = std::time( nullptr );
    for ( int i = 0; i < MAX_ATTEMPTS; ++i )
    {
        auto folder = tmp / ( "MeshInspectorScene" + std::to_string( t0 + i ) );
        if ( create_directories( folder, ec ) )
        {
            folder_ = std::move( folder );
            spdlog::info( "Temporary folder created: {}", utf8string( folder_ ) );
            break;
        }
    }
    if ( folder_.empty() )
        spdlog::error( "Failed to create unique temporary folder" );
}

UniqueTemporaryFolder::~UniqueTemporaryFolder()
{
    if ( folder_.empty() )
        return;
    MR_TIMER;
    if ( onPreTempFolderDelete_ )
        onPreTempFolderDelete_( folder_ );
    spdlog::info( "Deleting temporary folder: {}", utf8string( folder_ ) );
    std::error_code ec;
    if ( !std::filesystem::remove_all( folder_, ec ) )
    {
        spdlog::error( "Failed to remove folder: {}", ec.message() );
        return;
    }
}

const IOFilters SceneFileFilters =
{
    {"MeshInspector scene (.mru)","*.mru"}
};

tl::expected<Json::Value, std::string> deserializeJsonValue( const std::filesystem::path& path )
{
    if ( path.empty() )
        return tl::make_unexpected( "Cannot find parameters file" );

    std::ifstream ifs( path );
    if ( !ifs || ifs.bad() )
        return tl::make_unexpected( "Cannot open json file " + utf8string( path ) );

    std::string str( ( std::istreambuf_iterator<char>( ifs ) ),
                     std::istreambuf_iterator<char>() );

    if ( !ifs || ifs.bad() )
        return tl::make_unexpected( "Cannot read json file " + utf8string( path ) );

    ifs.close();

    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( !reader->parse( str.data(), str.data() + str.size(), &root, &error ) )
        return tl::make_unexpected( "Cannot parse json file: " + error );

    return root;
}

// path of item in filesystem, base - base path of scene root (%temp%/MeshInspectorScene)
tl::expected<void, std::string> compressOneItem( zip_t* archive, const std::filesystem::path& path, const std::filesystem::path& base,
    const std::vector<std::filesystem::path>& excludeFiles, const char * password )
{
    std::error_code ec;
    if ( std::filesystem::is_regular_file( path, ec ) )
    {
        auto excluded = std::find_if( excludeFiles.begin(), excludeFiles.end(), [&] ( const auto& a )
        {
            return std::filesystem::equivalent( a, path, ec );
        } );
        if ( excluded == excludeFiles.end() )
        {
            auto fileSource = zip_source_file( archive, utf8string( path ).c_str(), 0, 0 );
            if ( !fileSource )
                return tl::make_unexpected( "Cannot open file " + utf8string( path ) + " for reading" );

            auto archiveFilePath = utf8string( std::filesystem::relative( path, base, ec ) );
            // convert folder separators in Linux style for the latest 7-zip to open archive correctly
            std::replace( archiveFilePath.begin(), archiveFilePath.end(), '\\', '/' );
            const auto index = zip_file_add( archive, archiveFilePath.c_str(), fileSource, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8 );
            if ( index < 0 )
            {
                zip_source_free( fileSource );
                return tl::make_unexpected( "Cannot add file " + archiveFilePath + " to archive" );
            }

            if ( password )
            {
                if ( zip_file_set_encryption( archive, index, ZIP_EM_AES_256, password ) )
                    return tl::make_unexpected( "Cannot encrypt file " + archiveFilePath + " in archive" );
            }
        }
    }
    else
    {
        if ( !std::filesystem::is_directory( path, ec ) )
            return tl::make_unexpected( utf8string( path ) + " - is not file or directory." );
        if ( path != base )
        {
            auto archiveDirPath = utf8string( std::filesystem::relative( path, base, ec ) );
            // convert folder separators in Linux style for the latest 7-zip to open archive correctly
            std::replace( archiveDirPath.begin(), archiveDirPath.end(), '\\', '/' );
            if ( zip_dir_add( archive, archiveDirPath.c_str(), ZIP_FL_ENC_UTF_8 ) == -1 )
                return tl::make_unexpected( "Cannot add directory " + archiveDirPath + " to archive" );
        }
        for ( const auto& entry : std::filesystem::directory_iterator( path, ec ) )
        {
            auto res = compressOneItem( archive, entry.path(), base, excludeFiles, password );
            if ( !res.has_value() )
                return res;
        }
    }
    return {};
}

// this object stores a handle on open zip-archive, and automatically closes it in the destructor
class AutoCloseZip
{
public:
    AutoCloseZip( const char* path, int flags, int* err )
    {
        handle_ = zip_open( path, flags, err );
    }
    ~AutoCloseZip()
    {
        close();
    }
    operator zip_t *() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }
    int close()
    {
        if ( !handle_ )
            return 0;
        int res = zip_close( handle_ );
        handle_ = nullptr;
        return res;
    }

private:
    zip_t * handle_ = nullptr;
};

tl::expected<void, std::string> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder,
    const std::vector<std::filesystem::path>& excludeFiles, const char * password )
{
    MR_TIMER

    std::error_code ec;
    if ( !std::filesystem::is_directory( sourceFolder, ec ) )
        return tl::make_unexpected( "Directory '" + utf8string( sourceFolder ) + "' does not exist" );

    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err );
    if ( !zip )
        return tl::make_unexpected( "Cannot create zip, error code: " + std::to_string( err ) );

    auto res = compressOneItem( zip, sourceFolder, sourceFolder, excludeFiles, password );
    if ( !res.has_value() )
        return res;

    if ( zip.close() == -1 )
        return tl::make_unexpected( "Cannot close zip" );

    return res;
}

tl::expected<void, std::string> serializeMesh( const Mesh& mesh, const std::filesystem::path& path, const FaceBitSet* selection /*= nullptr */ )
{
    ObjectMesh obj;
    obj.setMesh( std::make_shared<Mesh>( mesh ) );
    if ( selection )
        obj.selectFaces( *selection );
    obj.setName( utf8string( path.stem() ) );
    return serializeObjectTree( obj, path );
}

tl::expected<void, std::string> decompressZip( const std::filesystem::path& zipFile, const std::filesystem::path& targetFolder, const char * password )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( targetFolder, ec ) )
        return tl::make_unexpected( "Directory does not exist " + utf8string( targetFolder ) );

    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_RDONLY, &err );
    if ( !zip )
        return tl::make_unexpected( "Cannot open zip, error code: " + std::to_string( err ) );

    if ( password )
        zip_set_default_password( zip, password );

    zip_stat_t stats;
    zip_file_t* zfile;
    std::vector<char> fileBufer;
    for ( int i = 0; i < zip_get_num_entries( zip, 0 ); ++i )
    {
        if ( zip_stat_index( zip, i, 0, &stats ) == -1 )
            return tl::make_unexpected( "Cannot process zip content" );

        std::string nameFixed = stats.name;
        std::replace( nameFixed.begin(), nameFixed.end(), '\\', '/' );
        std::filesystem::path relativeName = nameFixed;
        relativeName.make_preferred();
        std::filesystem::path newItemPath = targetFolder / relativeName;
        if ( !nameFixed.empty() && nameFixed.back() == '/' )
        {
            if ( !std::filesystem::exists( newItemPath.parent_path(), ec ) )
                if ( !std::filesystem::create_directories( newItemPath.parent_path(), ec ) )
                    return tl::make_unexpected( "Cannot create folder " + utf8string( newItemPath.parent_path() ) );
        }
        else
        {
            zfile = zip_fopen_index(zip,i,0);
            if ( !zfile )
                return tl::make_unexpected( "Cannot open zip file " + nameFixed );

            // in some manually created zip-files there is no folder entries for files in sub-folders;
            // so let us create directory each time before saving a file in it
            if ( !std::filesystem::exists( newItemPath.parent_path(), ec ) )
                if ( !std::filesystem::create_directories( newItemPath.parent_path(), ec ) )
                    return tl::make_unexpected( "Cannot create folder " + utf8string( newItemPath.parent_path() ) );

            std::ofstream ofs( newItemPath, std::ios::binary );
            if ( !ofs || ofs.bad() )
                return tl::make_unexpected( "Cannot create file " + utf8string( newItemPath ) );

            fileBufer.resize(stats.size);
            auto bitesRead = zip_fread(zfile,(void*)fileBufer.data(),fileBufer.size());
            if ( bitesRead != (zip_int64_t)stats.size )
                return tl::make_unexpected( "Cannot read file from zip " + nameFixed );

            zip_fclose(zfile);
            if ( !ofs.write( fileBufer.data(), fileBufer.size() ) )
                return tl::make_unexpected( "Cannot write file from zip " + utf8string( newItemPath ) );
            ofs.close();
        }
    }
    return {};
}

tl::expected<void, std::string> serializeObjectTree( const Object& object, const std::filesystem::path& path, 
    ProgressCallback progress, FolderCallback preCompress )
{
    MR_TIMER;
    if (path.empty())
        return tl::make_unexpected( "Cannot save to empty path" );

    UniqueTemporaryFolder scenePath( {} );
    if ( !scenePath )
        return tl::make_unexpected( "Cannot create temporary folder" );

    if ( progress && !progress( 0.0f ) )
        return tl::make_unexpected( "Canceled" );

    Json::Value root;
    root["FormatVersion"] = "0.0";
    auto saveModelFutures = object.serializeRecursive( scenePath, root, 0 );
    if ( !saveModelFutures.has_value() )
        return tl::make_unexpected( saveModelFutures.error() );

    auto paramsFile = scenePath / ( object.name() + ".json" );
    std::ofstream ofs( paramsFile );
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( !ofs || writer->write( root, &ofs ) != 0 )
        return tl::make_unexpected( "Cannot write parameters " + utf8string( paramsFile ) );

    ofs.close();

#ifdef __EMSCRIPTEN__
    for ( auto & f : saveModelFutures.value() )
        f.get();
#else
    if ( progress )
        progress( 0.1f );

    auto allSavesDone = [&]()
    {
        for ( auto & f : saveModelFutures.value() )
            if ( f.wait_for( std::chrono::milliseconds(200) ) == std::future_status::timeout )
                return false;
        return true;
    };
    auto numFinishedSaves = [&]()
    {
        int num = 0;
        for ( auto & f : saveModelFutures.value() )
            if ( f.wait_for( std::chrono::milliseconds(0) ) != std::future_status::timeout )
                ++num;
        return num;
    };

    // wait for all models are saved before making compressed folder
    while ( !allSavesDone() )
    {
        if ( progress )
        {
            progress( 0.1f + 0.8f * numFinishedSaves() / saveModelFutures.value().size() );
        }
    }

    if ( progress && !progress( 0.9f ) )
    {
        return tl::make_unexpected( "Canceled" );
    }
#endif
    if ( preCompress )
        preCompress( scenePath );

    auto res = compressZip( path, scenePath );

    if ( progress )
        progress( 1.0f );

    return res;
}

tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTree( const std::filesystem::path& path, FolderCallback postDecompress,
                                                                          ProgressCallback progressCb )
{
    MR_TIMER;
    UniqueTemporaryFolder scenePath( postDecompress );
    if ( !scenePath )
        return tl::make_unexpected( "Cannot create temporary folder" );
    auto res = decompressZip( path, scenePath );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );

    return deserializeObjectTreeFromFolder( scenePath, progressCb );
}

tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
                                                                                    ProgressCallback progressCb )
{
    MR_TIMER;

    std::error_code ec;
    std::filesystem::path jsonFile;
    for ( const auto& entry : std::filesystem::directory_iterator( folder, ec ) )
    {
        if ( entry.path().extension() == ".json" )
        {
            jsonFile = entry.path();
            break;
        }
    }

    auto readRes = deserializeJsonValue( jsonFile );
    if( !readRes.has_value() )
    {
        return tl::make_unexpected( readRes.error() );
    }
    auto root = readRes.value();

    auto typeTreeSize = root["Type"].size();
    std::shared_ptr<Object> rootObject;
    for (int i = typeTreeSize-1;i>=0;--i)
    {
        const auto& type = root["Type"][unsigned( i )];
        if ( type.isString() )
            rootObject = createObject( type.asString() );
        if ( rootObject )
            break;
    }
    if ( !rootObject )
        return tl::make_unexpected( "Unknown root object type" );

    int modelNumber{ 0 };
    int modelCounter{ 0 };
    if ( progressCb )
    {
        std::function<int( const Json::Value& )> calculateModelNum = [&calculateModelNum] ( const Json::Value& root )
        {
            int res{ 1 };

            if ( root["Children"].isNull() )
                return res;

            for ( const std::string& childKey : root["Children"].getMemberNames() )
            {
                if ( !root["Children"].isMember( childKey ) )
                    continue;

                const auto& child = root["Children"][childKey];
                if ( child.isNull() )
                    continue;
                res += calculateModelNum( child );
            }

            return res;
        };
        modelNumber = calculateModelNum( root );

        modelNumber = std::max( modelNumber, 1 );
        progressCb = [progressCb, &modelCounter, modelNumber] ( float v )
        {
            return progressCb( ( modelCounter + v ) / modelNumber );
        };
    }

    auto resDeser = rootObject->deserializeRecursive( folder, root, progressCb, &modelCounter );
    if ( !resDeser.has_value() )
        return tl::make_unexpected( "Cannot deserialize: " + resDeser.error());

    return rootObject;
}

void serializeToJson( const Vector2i& vec, Json::Value& root )
{
    root["x"] = vec.x;
    root["y"] = vec.y;
}

void serializeToJson( const Vector2f& vec, Json::Value& root )
{
    root["x"] = vec.x;
    root["y"] = vec.y;
}

void serializeToJson( const Vector3i& vec, Json::Value& root )
{
    root["x"] = vec.x;
    root["y"] = vec.y;
    root["z"] = vec.z;
}

void serializeToJson( const Vector3f& vec, Json::Value& root )
{
    root["x"] = vec.x;
    root["y"] = vec.y;
    root["z"] = vec.z;
}

void serializeToJson( const Vector4f& vec, Json::Value& root )
{
    root["x"] = vec.x;
    root["y"] = vec.y;
    root["z"] = vec.z;
    root["w"] = vec.w;
}

void serializeToJson( const Color& col, Json::Value& root )
{
    root["r"] = col.r;
    root["g"] = col.g;
    root["b"] = col.b;
    root["a"] = col.a;
}

void serializeToJson( const Matrix3f& matrix, Json::Value& root )
{
    serializeToJson( matrix.x, root["rowX"] );
    serializeToJson( matrix.y, root["rowY"] );
    serializeToJson( matrix.z, root["rowZ"] );
}

void serializeToJson( const AffineXf3f& xf, Json::Value& root )
{
    serializeToJson( xf.A, root["A"] );
    serializeToJson( xf.b, root["b"] );
}

void serializeToJson( const BitSet& bitset, Json::Value& root )
{
    std::vector<std::uint8_t> data;
    root["size"] = Json::UInt( bitset.size() );
    root["bits"] = encode64( (const std::uint8_t*) bitset.m_bits.data(), bitset.num_blocks() * sizeof( BitSet::block_type ) );
}

tl::expected<void, std::string> serializeToJson( const Mesh& mesh, Json::Value& root )
{
    std::ostringstream out;
    auto res = MeshSave::toPly( mesh, out );
    if ( res )
    {
        auto binString = out.str();
        root["ply"] = encode64( (const std::uint8_t*) binString.data(), binString.size() );
    }
    return res;
}

void serializeToJson( const Plane3f& plane, Json::Value& root )
{
    serializeToJson( plane.n, root["n"] );
    root["d"] = plane.d; 
}

void serializeToJson( const TriPointf& tp, Json::Value& root )
{
    root["a"] = tp.a; 
    root["b"] = tp.b; 
}

void serializeToJson( const MeshTriPoint& mtp, const MeshTopology & topology, Json::Value& root )
{
    auto canon = mtp.canonical( topology );
    serializeToJson( canon.bary, root );
    root["face"] = (int)topology.left( canon.e );
}

void serializeToJson( const PointOnFace& pf, Json::Value& root )
{
    root["face"] = (int)pf.face;
    serializeToJson( pf.point, root );
}

void deserializeFromJson( const Json::Value& root, Vector2i& vec )
{
    if ( root.isString() )
    {
        std::istringstream iss( root.asString() );
        iss >> vec;
    }
    else if ( root.isObject() && root["x"].isInt() && root["y"].isInt() )
    {
        vec.x = root["x"].asInt();
        vec.y = root["y"].asInt();
    }
}

void deserializeFromJson( const Json::Value& root, Vector2f& vec )
{
    if ( root.isString() )
    {
        std::istringstream iss( root.asString() );
        iss >> vec;
    }
    else if ( root.isObject() && root["x"].isNumeric() && root["y"].isNumeric() )
    {
        vec.x = root["x"].asFloat();
        vec.y = root["y"].asFloat();
    }
}

void deserializeFromJson( const Json::Value& root, Vector3i& vec )
{
    if ( root.isString() )
    {
        std::istringstream iss( root.asString() );
        iss >> vec;
    }
    else if ( root.isObject() && root["x"].isInt() && root["y"].isInt() && root["z"].isInt() )
    {
        vec.x = root["x"].asInt();
        vec.y = root["y"].asInt();
        vec.z = root["z"].asInt();
    }
}

void deserializeFromJson( const Json::Value& root, Vector3f& vec )
{
    if ( root.isString() )
    {
        std::istringstream iss( root.asString() );
        iss >> vec;
    }
    else if ( root.isObject() && root["x"].isNumeric() && root["y"].isNumeric() && root["z"].isNumeric() )
    {
        vec.x = root["x"].asFloat();
        vec.y = root["y"].asFloat();
        vec.z = root["z"].asFloat();
    }
}

void deserializeFromJson( const Json::Value& root, Vector4f& vec )
{
    if ( root.isString() )
    {
        std::istringstream iss( root.asString() );
        iss >> vec;
    }
    else if ( root.isObject() && root["x"].isNumeric() && root["y"].isNumeric() && root["z"].isNumeric() && root["w"].isNumeric() )
    {
        vec.x = root["x"].asFloat();
        vec.y = root["y"].asFloat();
        vec.z = root["z"].asFloat();
        vec.w = root["w"].asFloat();
    }
}

void deserializeFromJson( const Json::Value& root, Color& col )
{
    if ( root.isObject() && root["r"].isNumeric() && root["g"].isNumeric() && root["b"].isNumeric() && root["a"].isNumeric() )
    {
        col.r = uint8_t ( root["r"].asInt() );
        col.g = uint8_t ( root["g"].asInt() );
        col.b = uint8_t ( root["b"].asInt() );
        col.a = uint8_t ( root["a"].asInt() );
    }
}

void deserializeFromJson( const Json::Value& root, Matrix3f& matrix )
{
    deserializeFromJson( root["rowX"], matrix.x );
    deserializeFromJson( root["rowY"], matrix.y );
    deserializeFromJson( root["rowZ"], matrix.z );
}

void deserializeFromJson( const Json::Value& root, AffineXf3f& xf )
{
    if ( root["A"].isObject() )
        deserializeFromJson( root["A"], xf.A );
    deserializeFromJson( root["b"], xf.b );
}

void deserializeFromJson( const Json::Value& root, Plane3f& plane )
{
    deserializeFromJson( root["n"], plane.n );
    if ( root["d"].isNumeric() )
        plane.d = root["d"].asFloat();
}

void deserializeFromJson( const Json::Value& root, TriPointf& tp )
{
    if ( root["a"].isNumeric() )
        tp.a = root["a"].asFloat();
    if ( root["b"].isNumeric() )
        tp.b = root["b"].asFloat();
}

void deserializeFromJson( const Json::Value& root, MeshTriPoint& mtp, const MeshTopology& topology )
{
    deserializeFromJson( root, mtp.bary );
    if ( root["face"].isNumeric() )
        mtp.e = topology.edgeWithLeft( FaceId{ root["face"].asInt() } );
}

void deserializeFromJson( const Json::Value& root, PointOnFace& pf )
{
    if ( root["face"].isNumeric() )
        pf.face = FaceId{ root["face"].asInt() };
    deserializeFromJson( root, pf.point );
}

void deserializeFromJson( const Json::Value& root, BitSet& bitset )
{
    if ( root.isString() )
    {
        // old wide format
        std::istringstream iss( root.asString() );
        iss >> bitset;
    }
    else if ( root.isObject() && root["size"].isNumeric() && root["bits"].isString() )
    {
        // compact base64 format
        bitset.clear();
        bitset.resize( root["size"].asInt() );
        auto bin = decode64( root["bits"].asString() );
        auto bytes = std::min( bin.size(), bitset.num_blocks() * sizeof( BitSet::block_type ) );
        std::copy( bin.begin(), bin.begin() + bytes, (std::uint8_t*) bitset.m_bits.data() );
    }
}

tl::expected<Mesh, std::string> deserializeFromJson( const Json::Value& root, Vector<Color, VertId>* colors )
{
    if ( !root.isObject() )
        return tl::unexpected( std::string{ "deserialize mesh: json value is not an object" } );

    if ( !root["ply"].isString() )
        return tl::unexpected( std::string{ "deserialize mesh: json value does not have 'ply' string"} );
        
    auto bin = decode64( root["ply"].asString() );
    std::istringstream in( std::string( (const char *)bin.data(), bin.size() ) );
    return MeshLoad::fromPly( in, colors );
}

TEST( MRMesh, MeshToJson )
{
    Json::Value root;
    auto mesh = makeCube();
    auto saveRes = serializeToJson( mesh, root );
    ASSERT_TRUE( saveRes.has_value() );
    auto loadRes = deserializeFromJson( root );
    ASSERT_TRUE( loadRes.has_value() );
    auto mesh1 = std::move( loadRes.value() );
    ASSERT_EQ( mesh, mesh1 );
}

} // namespace MR
