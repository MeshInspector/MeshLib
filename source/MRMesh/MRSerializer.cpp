#include "MRSerializer.h"
#include "MRFile.h"
#include "MRObject.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix3.h"
#include "MRQuaternion.h"
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
#include "MRMeshBuilder.h"
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <zip.h>

#pragma warning( push )
#pragma warning( disable : 4062 4866 )
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#pragma warning( pop)

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif

#include <streambuf>
#include<stack>

namespace
{
using namespace MR;

struct MeshData
{
    std::shared_ptr<Mesh> mesh;
    Vector<Color, VertId> vertsColorMap;
    Vector<UVCoord, VertId> uvCoords;
    int materialIndex = -1;
};

struct Material
{
    Color baseColor;
    int textureIndex = -1;
};

std::vector<MeshTexture> readImages( const tinygltf::Model& model )
{
    std::vector<MeshTexture> result;
    result.reserve( model.images.size() );
    for ( auto& image : model.images )
    {
        result.emplace_back();
        auto& meshTexture = result.back();
        meshTexture.resolution = { image.width, image.height };
        meshTexture.pixels.resize( image.width * image.height );

        memcpy( meshTexture.pixels.data(), image.image.data(), image.width * image.height * sizeof( Color ) );
    }

    return result;
}

std::vector<Material> readMaterials( const tinygltf::Model& model )
{
    std::vector<Material> result;
    result.reserve( model.materials.size() );
    for ( auto& material : model.materials )
    {
        result.emplace_back();
        auto& curMaterial = result.back();

        auto colorIt = material.values.find( "baseColorFactor" );
        if ( colorIt != material.values.end() )
        {
            const auto& comps = colorIt->second.number_array;
            curMaterial.baseColor = Color( float( comps[0] ), float( comps[1] ), float( comps[2] ), float( comps[3] ) );
        }
        else
        {
            curMaterial.baseColor = Color::white();
        }

        curMaterial.textureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
    }

    return result;
}

tl::expected<int, std::string> readVertCoords( VertCoords& vertexCoordinates, const tinygltf::Model& model, const tinygltf::Primitive& primitive )
{
    if ( primitive.mode != TINYGLTF_MODE_TRIANGLES )
        return tl::make_unexpected( "This topology is not implemented" );

    auto posAttrib = primitive.attributes.find( "POSITION" );
    if ( posAttrib == primitive.attributes.end() )
        return tl::make_unexpected( "No vertex data" );

    auto accessor = model.accessors[posAttrib->second];
    auto bufferView = model.bufferViews[accessor.bufferView];
    auto buffer = model.buffers[bufferView.buffer];

    if ( accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || accessor.type != TINYGLTF_TYPE_VEC3 )
        return tl::make_unexpected( "This vertex component type is not implemented" );

    VertId start = VertId( vertexCoordinates.size() );
    vertexCoordinates.resize( vertexCoordinates.size() + accessor.count );
    memcpy( &vertexCoordinates[VertId( start )], &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof( Vector3f ) );

    return int( accessor.count );
}

void fillVertsColorMap( Vector<Color, VertId>& vertsColorMap, int vertexCount, const std::vector<Material>& materials, int materialIndex )
{
    const auto startPos = vertsColorMap.size();
    vertsColorMap.resize( vertsColorMap.size() + vertexCount );
    std::fill( ( uint32_t* )( vertsColorMap.data() + startPos ),
                ( uint32_t* )( vertsColorMap.data() + startPos + vertexCount ),
                materialIndex >= 0 ? materials[materialIndex].baseColor.getUInt32() : 0xFFFFFFFF );
}

std::string readUVCoords( Vector<UVCoord, VertId>& uvCoords, int vertexCount, const tinygltf::Model& model, const tinygltf::Primitive& primitive )
{
    uvCoords.resize( uvCoords.size() + vertexCount );

    auto posAttrib = primitive.attributes.find( "TEXCOORD_0" );
    if ( posAttrib == primitive.attributes.end() )
        return "";

    const auto& accessor = model.accessors[posAttrib->second];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    if ( ( accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT && accessor.componentType != TINYGLTF_COMPONENT_TYPE_DOUBLE ) || accessor.type != TINYGLTF_TYPE_VEC2 )
        return "Not implemented texcoord component type";

    if ( accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT )
    {
        memcpy( uvCoords.data() + uvCoords.size() - accessor.count, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof( Vector2f ) );
    }
    //TODO What if double
    return "";
}

std::string readTriangulation( Triangulation& t, const tinygltf::Model& model, const tinygltf::Primitive& primitive, VertId oldVertexCount, int vertexCount )
{
    if ( primitive.indices < 0 )
    {
        t.resize( t.size() + vertexCount / 3 );
        for ( int j = 0; j < vertexCount / 3; ++j )
            t.push_back( ThreeVertIds{ oldVertexCount + 3 * j,  oldVertexCount + 3 * j + 1, oldVertexCount + 3 * j + 2 } );

        return "";
    }

    const auto accessor = model.accessors[primitive.indices];
    const auto bufferView = model.bufferViews[accessor.bufferView];
    const auto buffer = model.buffers[bufferView.buffer];

    if ( accessor.componentType < TINYGLTF_COMPONENT_TYPE_BYTE || accessor.componentType > TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT || accessor.type != TINYGLTF_TYPE_SCALAR )
        return "Not implemented triangulation component type";

    const auto fillTriangles = [oldVertexCount]<typename ComponentType>( Triangulation & t, size_t triangleCount, ComponentType * pBuffer )
    {
        for ( size_t k = 0; k < triangleCount; ++k )
        {
            t.push_back( ThreeVertIds{ oldVertexCount + VertId( int( *pBuffer ) ), oldVertexCount + VertId( int( *( pBuffer + 1 ) ) ), oldVertexCount + VertId( int( *( pBuffer + 2 ) ) ) } );
            pBuffer += 3;
        }
    };

    auto pBuffer = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
    const size_t triangleCount = accessor.count / 3;

    switch ( accessor.componentType )
    {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        fillTriangles( t, triangleCount, ( int8_t* )pBuffer );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        fillTriangles( t, triangleCount, ( uint8_t* )pBuffer );
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        fillTriangles( t, triangleCount, ( int16_t* )pBuffer );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        fillTriangles( t, triangleCount, ( uint16_t* )pBuffer );
        break;
    case TINYGLTF_COMPONENT_TYPE_INT:
        fillTriangles( t, triangleCount, ( int32_t* )pBuffer );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        fillTriangles( t, triangleCount, ( uint32_t* )pBuffer );
        break;
    default:
        return "Not implemented triangulation component type";
    };

    return "";
}

tl::expected<std::vector<MeshData>, std::string> readMeshes( const tinygltf::Model& model, const std::vector<Material> materials, ProgressCallback callback )
{
    std::vector<MeshData> result;
    result.reserve( model.meshes.size() );

    for ( size_t i = 0; i < model.meshes.size(); ++i )
    {
        if ( !callback( 0.8f * ( i + 1 ) / float( model.meshes.size() ) ) )
            return tl::make_unexpected( "Operation was cancelled" );

        result.emplace_back();
        auto& meshData = result.back();

        const auto mesh = model.meshes[i];
        VertCoords vertexCoordinates;
        Triangulation t;

        for ( const auto& primitive : mesh.primitives )
        {
            VertId oldVertexCount = VertId( vertexCoordinates.size() );
            auto vertexCount = readVertCoords( vertexCoordinates, model, primitive );
            if ( !vertexCount.has_value() )
                return tl::make_unexpected( vertexCount.error() );

            fillVertsColorMap( meshData.vertsColorMap, *vertexCount, materials, primitive.material );

            if ( auto error = readUVCoords( meshData.uvCoords, *vertexCount, model, primitive ); !error.empty() )
                return tl::make_unexpected( error );

            if ( auto error = readTriangulation( t, model, primitive, oldVertexCount, *vertexCount ); !error.empty() )
                return tl::make_unexpected( error );

            if ( meshData.materialIndex < 0 )
                meshData.materialIndex = primitive.material;
        }

        std::vector<MeshBuilder::VertDuplication> dups;
        meshData.mesh = std::make_shared<Mesh>( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinates, t, &dups ) );
        if ( dups.empty() )
            continue;

        meshData.vertsColorMap.resize( meshData.vertsColorMap.size() + dups.size() );
        meshData.uvCoords.resize( meshData.uvCoords.size() + dups.size() );
        for ( const auto& dup : dups )
        {
            meshData.vertsColorMap[dup.dupVert] = meshData.vertsColorMap[dup.srcVert];
            meshData.uvCoords[dup.dupVert] = meshData.uvCoords[dup.srcVert];
        }
    }

    return result;
}

AffineXf3f readXf( const tinygltf::Node& node )
{

    if ( node.matrix.size() == 16 )
    {
        return AffineXf3f(
               Matrix3f{ { float( node.matrix[0] ),  float( node.matrix[4] ),  float( node.matrix[8] ) },
                          { float( node.matrix[1] ),  float( node.matrix[5] ),  float( node.matrix[9] ) },
                          { float( node.matrix[2] ),  float( node.matrix[6] ),  float( node.matrix[10] ) } },
               Vector3f{ float( node.matrix[12] ), float( node.matrix[13] ), float( node.matrix[14] ) } );
    }

    AffineXf3f xf;

    if ( node.translation.size() == 3 )
        xf = xf * AffineXf3f::translation( { float( node.translation[0] ), float( node.translation[1] ), float( node.translation[2] ) } );

    if ( node.rotation.size() == 4 )
        xf = xf * AffineXf3f( Matrix3f( Quaternion<float>( float( node.rotation[3] ), float( node.rotation[0] ), float( node.rotation[1] ), float( node.rotation[2] ) ) ), {} );

    if ( node.scale.size() == 3 )
        xf = xf * AffineXf3f( Matrix3f::scale( { float( node.scale[0] ), float( node.scale[1] ), float( node.scale[2] ) } ), {} );

    return xf;
}

}

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
    {"MeshInspector scene (.mru)","*.mru"},
    {"glTF scene (.glTF)","*.gltf"}
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
    ProgressCallback progressCb, FolderCallback preCompress )
{
    MR_TIMER;
    if (path.empty())
        return tl::make_unexpected( "Cannot save to empty path" );

    UniqueTemporaryFolder scenePath( {} );
    if ( !scenePath )
        return tl::make_unexpected( "Cannot create temporary folder" );

    if ( progressCb && !progressCb( 0.0f ) )
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
    reportProgress( progressCb, 0.1f );

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
        if ( progressCb )
        {
            progressCb( 0.1f + 0.8f * numFinishedSaves() / saveModelFutures.value().size() );
        }
    }

    if ( progressCb && !progressCb( 0.9f ) )
    {
        return tl::make_unexpected( "Canceled" );
    }
#endif
    if ( preCompress )
        preCompress( scenePath );

    auto res = compressZip( path, scenePath );

    reportProgress( progressCb, 1.0f );

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
    {
        std::string errorStr = resDeser.error();
        if ( errorStr != "Loading canceled" )
            errorStr = "Cannot deserialize: " + errorStr;
        return tl::make_unexpected( errorStr );
    }

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

void serializeToJson( const Matrix3f& matrix, Json::Value& root, bool skipIdentity )
{
    if ( skipIdentity && matrix == Matrix3f() )
        return; // skip saving, it will initialized as Matrix3f() anyway
    serializeToJson( matrix.x, root["rowX"] );
    serializeToJson( matrix.y, root["rowY"] );
    serializeToJson( matrix.z, root["rowZ"] );
}

void serializeToJson( const AffineXf3f& xf, Json::Value& root, bool skipIdentity )
{
    if ( skipIdentity && xf == AffineXf3f() )
        return; // skip saving, it will initialized as AffineXf3f() anyway
    serializeToJson( xf.A, root["A"] );
    serializeToJson( xf.b, root["b"] );
}

void serializeToJson( const BitSet& bitset, Json::Value& root )
{
    std::vector<std::uint8_t> data;
    root["size"] = Json::UInt( bitset.size() );
    root["bits"] = encode64( (const std::uint8_t*) bitset.m_bits.data(), bitset.num_blocks() * sizeof( BitSet::block_type ) );
}

void serializeViaVerticesToJson( const UndirectedEdgeBitSet& edges, const MeshTopology & topology, Json::Value& root )
{
    std::vector<VertId> verts;
    verts.reserve( edges.count() * 2 );
    for ( EdgeId e : edges )
    {
        auto o = topology.org( e );
        auto d = topology.dest( e );
        if ( o && d )
        {
            verts.push_back( o );
            verts.push_back( d );
        }
    }
    static_assert( sizeof( VertId ) == 4 );
    root["size"] = Json::UInt( edges.size() );
    root["vertpairs"] = encode64( (const std::uint8_t*) verts.data(), verts.size() * 4 );
}

void deserializeViaVerticesFromJson( const Json::Value& root, UndirectedEdgeBitSet& edges, const MeshTopology & topology )
{
    if ( !root.isObject() || !root["size"].isNumeric() || !root["vertpairs"].isString() )
    {
        deserializeFromJson( root, edges ); // deserialize from old format
        return;
    }

    edges.clear();
    edges.resize( root["size"].asInt() );
    auto bin = decode64( root["vertpairs"].asString() );

    for ( size_t i = 0; i + 8 < bin.size(); i += 8 )
    {
        VertId o, d;
        static_assert( sizeof( VertId ) == 4 );
        memcpy( &o, bin.data() + i, 4 );
        memcpy( &d, bin.data() + i + 4, 4 );
        auto e = topology.findEdge( o, d );
        if ( e && e.undirected() < edges.size() )
            edges.set( e.undirected() );
    }
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


tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback )
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    loader.LoadASCIIFromFile( &model, &err, &warn, asString( file.u8string() ) );

    if ( !err.empty() )
        return tl::make_unexpected( err );

    if ( model.meshes.empty() )
        return tl::make_unexpected( "No mesh in file" );

    auto textures = readImages( model );
    auto materials = readMaterials( model );
    auto meshesData = readMeshes( model, materials, callback );

    if ( !meshesData.has_value() )
        return tl::make_unexpected( meshesData.error() );

    std::shared_ptr<Object> scene = std::make_shared<Object>();
    std::shared_ptr<Object> rootObject;

    std::stack<int> nodeStack;
    std::stack<std::shared_ptr<Object>> objectStack;

    int counter = 0;
    int groupCounter = 0;
    int meshCounter = 0;

    for ( const auto nodeIndex : model.scenes[0].nodes )
    {
        if ( model.nodes[nodeIndex].mesh < 0 && model.nodes[nodeIndex].children.empty() )
            continue;

        nodeStack.push( nodeIndex );
        if ( model.nodes[nodeIndex].mesh >= 0 )
            objectStack.push( std::make_shared<ObjectMesh>() );
        else
            objectStack.push( std::make_shared<Object>() );

        rootObject = objectStack.top();

        while ( !nodeStack.empty() )
        {
            if ( !callback( 0.8f + 0.2f * ( ++counter ) / float( model.nodes.size() ) ) )
                return tl::make_unexpected( "Operation was cancelled" );

            const auto& node = model.nodes[nodeStack.top()];
            nodeStack.pop();
            auto curObject = objectStack.top();
            objectStack.pop();

            curObject->setXf( readXf( node ) );
            curObject->setName( node.name );

            if ( node.mesh >= 0 )
            {
                const auto& meshData = ( *meshesData )[node.mesh];

                auto objectMesh = std::static_pointer_cast< ObjectMesh >( curObject );
                objectMesh->setMesh( meshData.mesh );

                if ( auto textureIndex = materials[meshData.materialIndex].textureIndex; textureIndex >= 0 )
                {
                    objectMesh->setUVCoords( meshData.uvCoords );
                    objectMesh->setTexture( textures[textureIndex] );
                    objectMesh->setVisualizeProperty( true, VisualizeMaskType::Texture, ViewportMask::all() );
                }
                else
                {
                    objectMesh->setColoringType( ColoringType::VertsColorMap );
                    objectMesh->setVertsColorMap( meshData.vertsColorMap );
                }

                if ( node.name.empty() )
                    curObject->setName( model.meshes[node.mesh].name );
            }
            else
            {
                curObject->setAncillary( true );
            }

            if ( curObject->name().empty() )
            {
                curObject->setName( node.mesh >= 0 ? std::string( "Mesh_" ) + std::to_string( meshCounter++ ) :
                    std::string( "Group_" ) + std::to_string( groupCounter++ ) );
            }

            for ( auto& childNode : node.children )
            {
                if ( model.nodes[childNode].mesh < 0 && model.nodes[childNode].children.empty() )
                    continue;

                nodeStack.push( childNode );
                if ( model.nodes[childNode].mesh >= 0 )
                    objectStack.push( std::make_shared<ObjectMesh>() );
                else
                    objectStack.push( std::make_shared<Object>() );

                curObject->addChild( objectStack.top() );
            }
        }

        scene->addChild( rootObject );
    }

    return scene;
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
