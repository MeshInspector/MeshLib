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
#include "MRMeshTriPoint.h"
#include "MRMesh.h"
#include "MRStreamOperators.h"
#include "MRStringConvert.h"
#include "MRMeshTexture.h"
#include "MRDirectory.h"
#include "MRMeshLoad.h"
#include "MRMeshSave.h"
#include "MRObjectMesh.h"
#include "MRObjectSave.h"
#include "MRIOParsing.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"

#include <streambuf>

namespace MR
{

Expected<std::string> serializeJsonValue( const Json::Value& root )
{
    std::ostringstream oss;
    return serializeJsonValue( root, oss )
        .transform( [&] { return std::move( oss ).str(); } );
}

Expected<void> serializeJsonValue( const Json::Value& root, std::ostream& out )
{
    Json::StreamWriterBuilder builder;
    // see json/writer.h for available configurations
    std::unique_ptr<Json::StreamWriter> writer { builder.newStreamWriter() };

    if ( !out || writer->write( root, &out ) != 0 )
        // TODO: get an error string from the writer
        return unexpected( "Failed to write JSON" );

    return {};
}

Expected<void> serializeJsonValue( const Json::Value& root, const std::filesystem::path& path )
{
    // although json is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( path, std::ofstream::binary );
    return serializeJsonValue( root, out );
}

Expected<Json::Value> deserializeJsonValue( const char* data, size_t size )
{
    Timer t( "deserializeJsonValue( const std::string& )" );

    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( !reader->parse( data, data + size, &root, &error ) )
        return unexpected( "Cannot parse json file: " + error );

    return root;
}

Expected<Json::Value> deserializeJsonValue( const std::string& str )
{
    return deserializeJsonValue( str.data(), str.size() );
}

Expected<Json::Value> deserializeJsonValue( std::istream& in )
{
    Timer t( "deserializeJsonValue( std::istream& )" );

    auto maybeStr = readString( in );
    if ( !maybeStr )
        return unexpected( "Json " + maybeStr.error() );

    return deserializeJsonValue( *maybeStr );
}

Expected<Json::Value> deserializeJsonValue( const std::filesystem::path& path )
{
    if ( path.empty() )
        return unexpected( "Cannot find parameters file" );

    std::ifstream ifs( path, std::ifstream::binary );
    if ( !ifs || ifs.bad() )
        return unexpected( "Cannot open json file " + utf8string( path ) );

    return addFileNameInError( deserializeJsonValue( ifs ), path );
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

void serializeToJson( const Box3i& box, Json::Value& root )
{
    serializeToJson( box.min, root["min"] );
    serializeToJson( box.max, root["max"] );
}

void serializeToJson( const Box3f& box, Json::Value& root )
{
    serializeToJson( box.min, root["min"] );
    serializeToJson( box.max, root["max"] );
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
    root["bits"] = encode64( (const std::uint8_t*) bitset.bits().data(), bitset.num_blocks() * sizeof( BitSet::block_type ) );
}

void serializeToJson( const MeshTexture& texture, Json::Value& root )
{
    Timer t( "serializeToJson( const MeshTexture& )" );
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

void serializeToJson( const std::vector<TextureId>& texturePerFace, Json::Value& root )
{
    if ( texturePerFace.empty() )
        return;
    root["Size"] = int( texturePerFace.size() );
    root["Data"] = encode64( ( const uint8_t* )texturePerFace.data(), texturePerFace.size() * sizeof( TextureId ) );
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
    MR_TIMER;
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
    root["size"] = Json::UInt( edges.size() ); // saved for old versions of software before 1st July 2024
    root["vertpairs"] = encode64( (const std::uint8_t*) verts.data(), verts.size() * 4 );
}

void deserializeViaVerticesFromJson( const Json::Value& root, UndirectedEdgeBitSet& edges, const MeshTopology & topology )
{
    if ( !root.isObject() || !root["vertpairs"].isString() )
    {
        deserializeFromJson( root, edges ); // deserialize from old format
        return;
    }

    MR_TIMER;
    edges.clear();
    // not edges.resize( root["size"].asInt() ), because edge ids can change after loading mesh from CTM
    edges.resize( topology.undirectedEdgeSize() );
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
        std::copy( bin.begin(), bin.begin() + bytes, (std::uint8_t*) bitset.bits().data() );
    }
}

void deserializeFromJson( const Json::Value& root, MeshTexture& texture )
{
    Timer t( "deserializeFromJson( MeshTexture& )" );
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

void deserializeFromJson( const Json::Value& root, std::vector<TextureId>& texturePerFace )
{
    if ( root["Data"].isString() && root["Size"].isInt() )
    {
        const auto bin = decode64( root["Data"].asString() );
        const auto size = std::min<size_t>( root["Size"].asUInt64(), bin.size() / sizeof( TextureId ) );
        texturePerFace.resize( size );
        std::copy( ( TextureId* )bin.data(), ( TextureId* )( bin.data() ) + size, texturePerFace.data() );
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

Expected<void> serializeToJson( const Mesh& mesh, Json::Value& root )
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

Expected<Mesh> deserializeFromJson( const Json::Value& root, VertColors* colors )
{
    if ( !root.isObject() )
        return unexpected( std::string{ "deserialize mesh: json value is not an object" } );

    if ( !root["ply"].isString() )
        return unexpected( std::string{ "deserialize mesh: json value does not have 'ply' string"} );

    auto bin = decode64( root["ply"].asString() );
    std::istringstream in( std::string( (const char *)bin.data(), bin.size() ) );
    return MeshLoad::fromPly( in, { .colors = colors } );
}

Expected<void> serializeMesh( const Mesh& mesh, const std::filesystem::path& path, const FaceBitSet* selection, const char * serializeFormat )
{
    ObjectMesh obj;
    obj.setSerializeFormat( serializeFormat );
    obj.setMesh( std::make_shared<Mesh>( mesh ) );
    if ( selection )
        obj.selectFaces( *selection );
    obj.setName( utf8string( path.stem() ) );
    return serializeObjectTree( obj, path );
}

} // namespace MR
