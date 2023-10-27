#include "MRSerializer.h"
#include "MRFile.h"
#include "MRObject.h"
#include "MRVector3.h"
#include "MRVector4.h"
#include "MRMatrix2.h"
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
#include "MRGTest.h"
#include "MRMeshTexture.h"
#include "MRDirectory.h"
#include "MRZip.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"

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
        spdlog::error( "Cannot get temporary directory: {}", systemToUtf8( ec.message() ) );
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
        spdlog::error( "Failed to remove folder: {}", systemToUtf8( ec.message() ) );
        return;
    }
}

const IOFilters SceneFileFilters =
{
    {"MeshInspector scene (.mru)","*.mru"},
#ifndef MRMESH_NO_GLTF
    {"glTF JSON scene (.gltf)","*.gltf"},
    {"glTF binary scene (.glb)","*.glb"},
#endif
#ifndef MRMESH_NO_OPENCASCADE
    { "STEP model (.step,.stp)", "*.step;*.stp" },
#endif
    { "ZIP files (.zip)","*.zip" },
};

Expected<Json::Value, std::string> deserializeJsonValue( const std::string& str )
{
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( !reader->parse( str.data(), str.data() + str.size(), &root, &error ) )
        return unexpected( "Cannot parse json file: " + error );

    return root;
}

Expected<Json::Value, std::string> deserializeJsonValue( const std::filesystem::path& path )
{
    if ( path.empty() )
        return unexpected( "Cannot find parameters file" );

    std::ifstream ifs( path );
    if ( !ifs || ifs.bad() )
        return unexpected( "Cannot open json file " + utf8string( path ) );

    std::string str( ( std::istreambuf_iterator<char>( ifs ) ),
                     std::istreambuf_iterator<char>() );

    if ( !ifs || ifs.bad() )
        return unexpected( "Cannot read json file " + utf8string( path ) );

    ifs.close();

    return deserializeJsonValue( str );
}

VoidOrErrStr serializeMesh( const Mesh& mesh, const std::filesystem::path& path, const FaceBitSet* selection /*= nullptr */ )
{
    ObjectMesh obj;
    obj.setMesh( std::make_shared<Mesh>( mesh ) );
    if ( selection )
        obj.selectFaces( *selection );
    obj.setName( utf8string( path.stem() ) );
    return serializeObjectTree( obj, path );
}

VoidOrErrStr serializeObjectTree( const Object& object, const std::filesystem::path& path, 
    ProgressCallback progressCb, FolderCallback preCompress )
{
    MR_TIMER;
    if (path.empty())
        return unexpected( "Cannot save to empty path" );

    UniqueTemporaryFolder scenePath( {} );
    if ( !scenePath )
        return unexpected( "Cannot create temporary folder" );

    if ( progressCb && !progressCb( 0.0f ) )
        return unexpected( "Canceled" );

    Json::Value root;
    root["FormatVersion"] = "0.0";
    auto saveModelFutures = object.serializeRecursive( scenePath, root, 0 );
    if ( !saveModelFutures.has_value() )
        return unexpected( saveModelFutures.error() );

    auto paramsFile = scenePath / ( object.name() + ".json" );
    std::ofstream ofs( paramsFile );
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( !ofs || writer->write( root, &ofs ) != 0 )
        return unexpected( "Cannot write parameters " + utf8string( paramsFile ) );

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

    if ( !reportProgress( progressCb, 0.9f ) )
        return unexpectedOperationCanceled();

#endif
    if ( preCompress )
        preCompress( scenePath );

    return compressZip( path, scenePath, {}, nullptr, subprogress( progressCb, 0.9f, 1.0f ) );
}

Expected<std::shared_ptr<Object>, std::string> deserializeObjectTree( const std::filesystem::path& path, FolderCallback postDecompress,
                                                                          ProgressCallback progressCb )
{
    MR_TIMER;
    UniqueTemporaryFolder scenePath( postDecompress );
    if ( !scenePath )
        return unexpected( "Cannot create temporary folder" );
    auto res = decompressZip( path, scenePath );
    if ( !res.has_value() )
        return unexpected( res.error() );

    return deserializeObjectTreeFromFolder( scenePath, progressCb );
}

Expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
                                                                                    ProgressCallback progressCb )
{
    MR_TIMER;

    std::error_code ec;
    std::filesystem::path jsonFile;
    for ( auto entry : Directory{ folder, ec } )
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
        return unexpected( readRes.error() );
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
        return unexpected( "Unknown root object type" );

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
        return unexpected( errorStr );
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

void serializeToJson( const Matrix2f& matrix, Json::Value& root, bool skipIdentity )
{
    if ( skipIdentity && matrix == Matrix2f() )
        return; // skip saving, it will initialized as Matrix2f() anyway
    serializeToJson( matrix.x, root["rowX"] );
    serializeToJson( matrix.y, root["rowY"] );
}

void serializeToJson( const Matrix3f& matrix, Json::Value& root, bool skipIdentity )
{
    if ( skipIdentity && matrix == Matrix3f() )
        return; // skip saving, it will initialized as Matrix3f() anyway
    serializeToJson( matrix.x, root["rowX"] );
    serializeToJson( matrix.y, root["rowY"] );
    serializeToJson( matrix.z, root["rowZ"] );
}

void serializeToJson( const AffineXf2f& xf, Json::Value& root, bool skipIdentity )
{
    if ( skipIdentity && xf == AffineXf2f() )
        return; // skip saving, it will initialized as AffineXf2f() anyway
    serializeToJson( xf.A, root["A"] );
    serializeToJson( xf.b, root["b"] );
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

void serializeToJson( const MeshTexture& texture, Json::Value& root )
{
    switch ( texture.filter )
    {
    case FilterType::Linear:
        root["FilterType"] = "Linear";
        break;
    case FilterType::Discrete:
        root["FilterType"] = "Discrete";
        break;
    default:
        assert( false );
        root["FilterType"] = "Unknown";
        break;
    }

    switch ( texture.wrap )
    {
    case WrapType::Clamp:
        root["WrapType"] = "Clamp";
        break;
    case WrapType::Mirror :
        root["WrapType"] = "Mirror";
        break;
    case WrapType::Repeat:
        root["WrapType"] = "Repeat";
        break;
    default:
        assert( false );
        root["WrapType"] = "Unknown";
        break;
    }

    serializeToJson( texture.resolution, root["Resolution"] );
    root["Data"] = encode64( ( const uint8_t* )texture.pixels.data(), texture.pixels.size() * sizeof( Color ) );
}

void serializeToJson( const std::vector<UVCoord>& uvCoords, Json::Value& root )
{
    if ( uvCoords.empty() )
        return;
    root["Size"] = int( uvCoords.size() );
    root["Data"] = encode64( ( const uint8_t* )uvCoords.data(), uvCoords.size() * sizeof( UVCoord ) );
}

void serializeToJson( const std::vector<Color>& colors, Json::Value& root )
{
    if ( colors.empty() )
        return;
    root["Size"] = int( colors.size() );
    root["Data"] = encode64( ( const uint8_t* )colors.data(), colors.size() * sizeof( Color ) );
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

    for ( size_t i = 0; i + 8 <= bin.size(); i += 8 )
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

VoidOrErrStr serializeToJson( const Mesh& mesh, Json::Value& root )
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

void deserializeFromJson( const Json::Value& root, Matrix2f& matrix )
{
    deserializeFromJson( root["rowX"], matrix.x );
    deserializeFromJson( root["rowY"], matrix.y );
}

void deserializeFromJson( const Json::Value& root, Matrix3f& matrix )
{
    deserializeFromJson( root["rowX"], matrix.x );
    deserializeFromJson( root["rowY"], matrix.y );
    deserializeFromJson( root["rowZ"], matrix.z );
}

void deserializeFromJson( const Json::Value& root, AffineXf2f& xf )
{
    if ( root["A"].isObject() )
        deserializeFromJson( root["A"], xf.A );
    deserializeFromJson( root["b"], xf.b );
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

Expected<Mesh, std::string> deserializeFromJson( const Json::Value& root, VertColors* colors )
{
    if ( !root.isObject() )
        return unexpected( std::string{ "deserialize mesh: json value is not an object" } );

    if ( !root["ply"].isString() )
        return unexpected( std::string{ "deserialize mesh: json value does not have 'ply' string"} );
        
    auto bin = decode64( root["ply"].asString() );
    std::istringstream in( std::string( (const char *)bin.data(), bin.size() ) );
    return MeshLoad::fromPly( in, { .colors = colors } );
}

void deserializeFromJson( const Json::Value& root, MeshTexture& texture )
{
    if ( root["FilterType"].isString() )
    {
        auto filterName = root["FilterType"].asString();
        if ( filterName == "Linear" )
            texture.filter = FilterType::Linear;
        else if ( filterName == "Discrete" )
            texture.filter = FilterType::Discrete;
    }

    if ( root["WrapType"].isString() )
    {
        auto wrapName = root["WrapType"].asString();
        if ( wrapName == "Clamp" )
            texture.wrap = WrapType::Clamp;
        else if ( wrapName == "Mirror" )
            texture.wrap = WrapType::Mirror;
        else if ( wrapName == "Repeat" )
            texture.wrap = WrapType::Repeat;
    }
    deserializeFromJson( root["Resolution"], texture.resolution );
    if ( root["Data"].isString() )
    {
        texture.pixels.resize( texture.resolution.x * texture.resolution.y );
        auto bin = decode64( root["Data"].asString() );
        auto numColors = std::min( bin.size() / sizeof( Color ), texture.pixels.size() );
        std::copy( ( Color* )bin.data(), ( Color* )( bin.data() ) + numColors, texture.pixels.data() );
    }
}

void deserializeFromJson( const Json::Value& root, std::vector<UVCoord>& uvCoords )
{
    if ( root["Data"].isString() && root["Size"].isInt() )
    {
        const auto bin = decode64( root["Data"].asString() );
        const auto size = std::min<size_t>( root["Size"].asUInt64(), bin.size() / sizeof( UVCoord ) );
        uvCoords.resize( size );
        std::copy( ( UVCoord* )bin.data(), ( UVCoord* )( bin.data() ) + size, uvCoords.data() );
    }
}

void deserializeFromJson( const Json::Value& root, std::vector<Color>& colors )
{
    if ( root["Data"].isString() && root["Size"].isUInt64() )
    {
        const auto bin = decode64( root["Data"].asString() );
        const auto size = std::min<size_t>( root["Size"].asUInt64(), bin.size() / sizeof( Color ) );
        colors.resize( size );
        std::copy( ( Color* )bin.data(), ( Color* )( bin.data() ) + size, colors.data() );
    }
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
