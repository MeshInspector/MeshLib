#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRVoxelsLoad.h"
#include "MRStringConvert.h"
#include "MRPch/MRJson.h"
#include <fstream>

namespace MR
{

namespace VoxelsLoad
{

Expected<VdbVolume, std::string> fromGav( const std::filesystem::path& file, const ProgressCallback& cb )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
    return addFileNameInError( fromGav( in, cb ), file );
}

Expected<VdbVolume, std::string> fromGav( std::istream& in, const ProgressCallback& cb )
{
    uint32_t headerLen = 0;
    if ( !in.read( (char*) &headerLen, sizeof( headerLen ) ) )
        return unexpected( "Gav-header size read error" );

    std::string header;
    header.resize( headerLen );
    if ( !in.read( header.data(), headerLen ) )
        return unexpected( "Gav-header read error" );

    Json::Value headerJson;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( !reader->parse( header.data(), header.data() + header.size(), &headerJson, &error ) )
        return unexpected( "Gav-header parse error: " + error );

    RawParameters params;

    if ( !headerJson["ValueType"].isString() )
        return unexpected( "Gav-header misses ValueType" );

    const std::string valueType = headerJson["ValueType"].asString();
    if ( valueType == "UChar" )
        params.scalarType = RawParameters::ScalarType::UInt8;
    else if( valueType == "UInt16" )
        params.scalarType = RawParameters::ScalarType::UInt16;
    else if ( valueType == "UInt32" )
        params.scalarType = RawParameters::ScalarType::UInt32;
    else if ( valueType == "Char" )
        params.scalarType = RawParameters::ScalarType::Int8;
    else if ( valueType == "Int16" )
        params.scalarType = RawParameters::ScalarType::Int16;
    else if ( valueType == "Int32" )
        params.scalarType = RawParameters::ScalarType::Int32;
    else if ( valueType == "Float" )
        params.scalarType = RawParameters::ScalarType::Float32;
    else
        return unexpected( "Gav-header ValueType has unknown value: " + valueType );

    auto dimJson = headerJson["Dimensions"];
    if ( !dimJson.isObject() || !dimJson["X"].isInt() || !dimJson["Y"].isInt() || !dimJson["Z"].isInt() )
        return unexpected( "Gav-header misses Dimensions" );

    params.dimensions.x = dimJson["X"].asInt();
    params.dimensions.y = dimJson["Y"].asInt();
    params.dimensions.z = dimJson["Z"].asInt();

    auto voxJson = headerJson["VoxelSize"];
    if ( !voxJson.isObject() || !voxJson["X"].isNumeric() || !voxJson["Y"].isNumeric() || !voxJson["Z"].isNumeric() )
        return unexpected( "Gav-header misses VoxelSize" );

    params.voxelSize.x = voxJson["X"].asFloat();
    params.voxelSize.y = voxJson["Y"].asFloat();
    params.voxelSize.z = voxJson["Z"].asFloat();

    if ( headerJson["Compression"].isString() )
        return unexpected( "Compressed Gav-files are not supported" );

    return fromRaw( in, params, cb );
}

} //namespace VoxelsLoad

} //namespace MR

#endif // !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXELS )
