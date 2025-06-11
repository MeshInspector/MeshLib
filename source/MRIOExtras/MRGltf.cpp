#include "MRGltf.h"
#ifndef MRIOEXTRAS_NO_GLTF
#include "MRMesh/MRVector.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRQuaternion.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRParallelFor.h"
#include "MRPch/MRSuppressWarning.h"

MR_SUPPRESS_WARNING_PUSH

#if (defined(__APPLE__) && defined(__clang__)) || __EMSCRIPTEN__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if __EMSCRIPTEN__
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif

#pragma warning( disable : 4018 ) //'>=': signed/unsigned mismatch
#pragma warning( disable : 4062 ) //enumerator 'nlohmann::json_abi_v3_11_2::detail::value_t::binary' in switch of enum 'nlohmann::json_abi_v3_11_2::detail::value_t' is not handled
#pragma warning( disable : 4242 ) //'argument': conversion from 'int' to 'short', possible loss of data
#pragma warning( disable : 4244 ) //'argument': conversion from 'int' to 'short', possible loss of data
#pragma warning( disable : 4267 ) //'argument': conversion from 'size_t' to 'int', possible loss of data
#pragma warning( disable : 4866 ) //compiler may not enforce left-to-right evaluation order for call to 'nlohmann::json_abi_v3_11_2::basic_json<std::map,std::vector,std::basic_string<char,std::char_traits<char>,std::allocator<char> >,bool,__int64,unsigned __int64,double,std::allocator,nlohmann::json_abi_v3_11_2::adl_serializer,std::vector<unsigned char,std::allocator<unsigned char> > >::operator[]'

#if __GNUC__ >= 14
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#if __GNUC__ >= 15
#pragma GCC diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#if __clang_major__ == 20
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

MR_SUPPRESS_WARNING_POP

#include <stack>

namespace
{
using namespace MR;

//holds together mesh itself, vertex colors, texture coordinates and index of material
struct MeshData
{
    std::shared_ptr<Mesh> mesh;
    VertColors vertsColorMap;
    VertUVCoords uvCoords;
    int materialIndex = -1;
};

//holds together base color and index of texture
struct Material
{
    Color baseColor;
    int textureIndex = -1;

    bool operator==( const Material& other ) const
    {
        return baseColor == other.baseColor && textureIndex == other.textureIndex;
    }
};

Expected<std::vector<MeshTexture>> readImages( const tinygltf::Model& model )
{
    std::vector<MeshTexture> result;
    result.reserve( model.images.size() );
    for ( auto& image : model.images )
    {
        if ( image.image.empty() )
            return unexpected( "Image file '" + image.uri + "' is missing" );

        auto& meshTexture = result.emplace_back();
        meshTexture.filter = FilterType::Linear;
        meshTexture.resolution = { image.width, image.height };
        meshTexture.pixels.resize( image.width * image.height );

        std::copy( image.image.data(), image.image.data() + image.width * image.height * sizeof( Color ), ( uint8_t* )meshTexture.pixels.data() );
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

Expected<int> readVertCoords( VertCoords& vertexCoordinates, const tinygltf::Model& model, const tinygltf::Primitive& primitive )
{
    if ( primitive.mode != TINYGLTF_MODE_TRIANGLES )
        return unexpected( "This topology is not implemented" );

    auto posAttrib = primitive.attributes.find( "POSITION" );
    if ( posAttrib == primitive.attributes.end() )
        return unexpected( "No vertex data" );

    if ( posAttrib->second >= model.accessors.size() )
        return unexpected( "Invalid accessor index" );
    const auto& accessor = model.accessors[posAttrib->second];

    if ( accessor.bufferView >= model.bufferViews.size() )
        return unexpected( "Invalid bufferView index" );
    const auto& bufferView = model.bufferViews[accessor.bufferView];

    if ( bufferView.buffer >= model.buffers.size() )
        return unexpected( "Invalid buffer index" );
    const auto& buffer = model.buffers[bufferView.buffer];

    if ( accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || accessor.type != TINYGLTF_TYPE_VEC3 )
        return unexpected( "This vertex component type is not implemented" );

    VertId start = VertId( vertexCoordinates.size() );
    vertexCoordinates.resize( vertexCoordinates.size() + accessor.count );

    if ( bufferView.byteStride == 0 )
    {
        std::copy( &buffer.data[accessor.byteOffset + bufferView.byteOffset],
            &buffer.data[accessor.byteOffset + bufferView.byteOffset + accessor.count * sizeof( Vector3f )],
            ( uint8_t* )&vertexCoordinates[VertId( start )] );
    }
    else
    {
        const auto startSpan = vertexCoordinates.vec_.begin() + size_t( start );
        ParallelFor( startSpan, vertexCoordinates.vec_.end(), [&] ( auto it )
        {
            const size_t i = std::distance( startSpan, it );
            *it = *( Vector3f* )( &buffer.data[accessor.byteOffset + bufferView.byteOffset + i * bufferView.byteStride] );
        } );
    }

    return int( accessor.count );
}

template<typename ChannelType>
constexpr float channelMax()
{
    if constexpr ( std::is_same_v<ChannelType, float> || std::is_same_v<ChannelType, double> )
        return 1.0f;
    else
        return float( std::numeric_limits<ChannelType>::max() );
}

Expected<void> fillVertsColorMap( VertColors& vertsColorMap, int vertexCount, const std::vector<Material>& materials, int materialIndex, const tinygltf::Model& model, const tinygltf::Primitive& primitive )
{
    const VertId startPos = VertId( vertsColorMap.size() );
    vertsColorMap.resize( vertsColorMap.size() + vertexCount );

    const auto posAttrib = primitive.attributes.find( "COLOR_0" );
    if ( posAttrib == primitive.attributes.end() )
    {
        std::fill( ( uint32_t* )( vertsColorMap.data() + startPos ),
                    ( uint32_t* )( vertsColorMap.data() + startPos + vertexCount ),
                    materialIndex >= 0 ? materials[materialIndex].baseColor.getUInt32() : 0xFFFFFFFF );

        return {};
    }

    const auto& accessor = model.accessors[posAttrib->second];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const auto fillColorMap = [&]<typename ChannelType, int channelCount>( VertColors & vertsColorMap )
    {
        const float cMax = channelMax<ChannelType>();

        ParallelFor( vertsColorMap, [&] ( VertId v )
        {
            if constexpr ( channelCount == 3 )
            {
                const Vector3<ChannelType> col = *( Vector3<ChannelType>* )( &buffer.data[accessor.byteOffset + bufferView.byteOffset + v * bufferView.byteStride] );
                vertsColorMap[startPos + v] = Color( float( col[0] / cMax ), float( col[1] / cMax ), float( col[2] / cMax ) );
            }
            else if constexpr ( channelCount == 4 )
            {
                const Vector4<ChannelType> col = *( Vector4<ChannelType>* )( &buffer.data[accessor.byteOffset + bufferView.byteOffset + v * bufferView.byteStride] );
                vertsColorMap[startPos + v] = Color( float( col[0] / cMax ), float( col[1] / cMax ), float( col[2] / cMax ), float( col[3] / cMax ) );
            }
        } );
    };

    if ( accessor.type != TINYGLTF_TYPE_VEC3 && accessor.type != TINYGLTF_TYPE_VEC4 )
        return unexpected( "This vertex color type is not supported" );

    switch ( accessor.componentType )
    {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < float, 3 > ( vertsColorMap ) : fillColorMap.operator() < float, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < double, 3 > ( vertsColorMap ) : fillColorMap.operator() < double, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < int8_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < int8_t, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < uint8_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < uint8_t, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < int16_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < int16_t, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < uint16_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < uint16_t, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_INT:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < int32_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < int32_t, 4 > ( vertsColorMap );
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        ( accessor.type == TINYGLTF_TYPE_VEC3 ) ? fillColorMap.operator() < uint32_t, 3 > ( vertsColorMap ) : fillColorMap.operator() < uint32_t, 4 > ( vertsColorMap );
        break;
    default:
        return unexpected( "This vertex color type is not supported" );
    }

    return{};
}

std::string readUVCoords( VertUVCoords& uvCoords, int vertexCount, const tinygltf::Model& model, const tinygltf::Primitive& primitive )
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
        std::copy( &buffer.data[accessor.byteOffset + bufferView.byteOffset], &buffer.data[accessor.byteOffset + bufferView.byteOffset] + accessor.count * sizeof( Vector2f ), ( uint8_t* )( uvCoords.data() + uvCoords.size() - accessor.count ) );
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

    const auto& accessor = model.accessors[primitive.indices];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

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

Expected<std::vector<MeshData>> readMeshes( const tinygltf::Model& model, const std::vector<Material> materials, ProgressCallback callback )
{
    std::vector<MeshData> result;
    result.reserve( model.meshes.size() );

    for ( size_t meshId = 0; meshId < model.meshes.size(); ++meshId )
    {
        if ( callback && !callback( 0.8f * ( meshId + 1 ) / float( model.meshes.size() ) ) )
            return unexpected( "Operation was cancelled" );

        result.emplace_back();
        auto& meshData = result.back();

        const auto mesh = model.meshes[meshId];
        VertCoords vertexCoordinates;
        Triangulation t;

        bool areMaterialsSame = true;

        for ( size_t primitiveId = 0; primitiveId < mesh.primitives.size(); ++primitiveId )
        {
            const auto& primitive = mesh.primitives[primitiveId];
            VertId oldVertexCount = VertId( vertexCoordinates.size() );
            auto vertexCount = readVertCoords( vertexCoordinates, model, primitive );
            if ( !vertexCount.has_value() )
                return unexpected( vertexCount.error() );

            if ( auto error = fillVertsColorMap( meshData.vertsColorMap, *vertexCount, materials, primitive.material, model, primitive ); !error.has_value() )
                return unexpected( error.error() );

            if ( auto error = readUVCoords( meshData.uvCoords, *vertexCount, model, primitive ); !error.empty() )
                return unexpected( error );

            if ( auto error = readTriangulation( t, model, primitive, oldVertexCount, *vertexCount ); !error.empty() )
                return unexpected( error );

            if ( meshData.materialIndex < 0 )
                meshData.materialIndex = primitive.material;

            areMaterialsSame &= ( primitive.material >= 0 );
            if ( primitiveId > 0 )
                areMaterialsSame &= ( primitive.material == mesh.primitives[primitiveId - 1].material );
        }

        if ( areMaterialsSame )
            meshData.vertsColorMap.clear();

        std::vector<MeshBuilder::VertDuplication> dups;
        meshData.mesh = std::make_shared<Mesh>( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinates, t, &dups ) );
        if ( dups.empty() )
            continue;

        if ( !areMaterialsSame )
        {
            meshData.vertsColorMap.resize( meshData.vertsColorMap.size() + dups.size() );
            for ( const auto& dup : dups )
            {
                meshData.vertsColorMap[dup.dupVert] = meshData.vertsColorMap[dup.srcVert];
            }
        }

        if ( meshData.materialIndex >= 0 && model.materials[meshData.materialIndex].pbrMetallicRoughness.baseColorTexture.index >= 0 )
        {
            meshData.uvCoords.resize( meshData.uvCoords.size() + dups.size() );
            for ( const auto& dup : dups )
            {
                meshData.uvCoords[dup.dupVert] = meshData.uvCoords[dup.srcVert];
            }
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

Expected<std::shared_ptr<Object>> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback )
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    if ( file.extension() == u8".gltf" )
        loader.LoadASCIIFromFile( &model, &err, &warn, asString( file.u8string() ) );
    else if ( file.extension() == u8".glb" )
        loader.LoadBinaryFromFile( &model, &err, &warn, asString( file.u8string() ) );

    if ( !err.empty() )
        return unexpected( err );

    if ( model.meshes.empty() )
        return unexpected( "No mesh in file" );

    auto texturesOrErr = readImages( model );
    if ( !texturesOrErr )
        return unexpected( texturesOrErr.error() );

    auto textures = std::move( *texturesOrErr );
    auto materials = readMaterials( model );
    auto meshesData = readMeshes( model, materials, callback );

    if ( !meshesData.has_value() )
        return unexpected( meshesData.error() );

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
            if ( callback && !callback( 0.8f + 0.2f * ( ++counter ) / float( model.nodes.size() ) ) )
                return unexpected( "Operation was cancelled" );

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

                if ( meshData.materialIndex >= 0 )
                {
                    const auto textureIndex = materials[meshData.materialIndex].textureIndex;
                    if ( textureIndex >= 0 && model.textures[textureIndex].source >= 0 )
                    {
                        objectMesh->setUVCoords( meshData.uvCoords );
                        objectMesh->setTextures( { textures[model.textures[textureIndex].source] } );
                        objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                    }
                    else if ( !meshData.vertsColorMap.empty() )
                    {
                        objectMesh->setColoringType( ColoringType::VertsColorMap );
                        objectMesh->setVertsColorMap( meshData.vertsColorMap );
                    }
                    else
                    {
                        objectMesh->setFrontColor( materials[meshData.materialIndex].baseColor, false );
                    }
                }
                else if ( !meshData.vertsColorMap.empty() )
                {
                    objectMesh->setColoringType( ColoringType::VertsColorMap );
                    objectMesh->setVertsColorMap( meshData.vertsColorMap );
                }

                if ( node.name.empty() )
                    curObject->setName( model.meshes[node.mesh].name );
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

Expected<void> serializeObjectTreeToGltf( const Object& root, const std::filesystem::path& file, ProgressCallback callback )
{
    tinygltf::Model model;
    model.asset.generator = "MeshLib";
    model.asset.version = "2.0";

    model.buffers.emplace_back();
    auto& buffer = model.buffers[0].data;

    model.scenes.emplace_back();

    const auto materialHash = [] ( const Material& material )
    {
        return std::hash<uint32_t>()( material.baseColor.getUInt32() ) ^ std::hash<int>()( material.textureIndex );
    };

    std::unordered_map<Material, int, decltype( materialHash )> materials( {}, materialHash );

    const auto textureHash = [] ( const MeshTexture& meshTexture )
    {
        auto res = std::hash<int>()( meshTexture.resolution.x ) ^ std::hash<int>()( meshTexture.resolution.y );
        for ( auto& pixel : meshTexture.pixels )
            res ^= std::hash<uint32_t>()( pixel.getUInt32() );

        return res;
    };

    const auto textureCompare = [] ( const MeshTexture& a, const MeshTexture& b )
    {
        return a.pixels == b.pixels && a.resolution == b.resolution;
    };

    std::unordered_map<MeshTexture, int, decltype( textureHash ), decltype( textureCompare )> textures( {}, textureHash, textureCompare );


    std::stack<std::shared_ptr<const Object>> objectStack;
    std::stack<size_t> indexStack;

    if ( callback && !callback( 0.1f ) )
        return unexpected( "Operation was cancelled" );

    for ( size_t childIndex = 0; childIndex < root.children().size(); ++childIndex )
    {
        if ( root.children()[childIndex]->isAncillary() )
            continue;

        objectStack.push( root.children()[childIndex] );
        size_t lastIndex = model.nodes.size();
        indexStack.push( lastIndex );
        model.scenes[0].nodes.push_back( int( lastIndex ) );

        while ( !objectStack.empty() )
        {
            auto curObj = objectStack.top();
            objectStack.pop();
            size_t curIndex = indexStack.top();
            indexStack.pop();

            if ( model.nodes.size() < curIndex + 1 )
                model.nodes.resize( curIndex + 1 );

            auto& curNode = model.nodes[curIndex];
            curNode.name = curObj->name();

            const auto& A = curObj->xf().A;
            const auto& b = curObj->xf().b;
            curNode.matrix = { A[0][0], A[1][0], A[2][0], 0,
                               A[0][1], A[1][1], A[2][1], 0,
                               A[0][2], A[1][2], A[2][2], 0,
                               b[0], b[1], b[2], 1 };

            auto curObjectMesh = curObj->asType<ObjectMesh>();
            if ( curObjectMesh && curObjectMesh->mesh() )
            {
                Material material;
                material.baseColor = curObjectMesh->getFrontColor( false );
                if ( const auto& vertsColorMap = curObjectMesh->getVertsColorMap(); !vertsColorMap.empty() )
                {
                    material.baseColor = vertsColorMap.front();
                }

                const auto& texture = curObjectMesh->getTexture();

                if ( texture.resolution.x > 0 && texture.resolution.y > 0 )
                {
                    const auto textureIt = textures.find( texture );
                    material.textureIndex = int( textures.size() );
                    if ( textureIt == textures.end() )
                    {
                        textures.insert_or_assign( texture, int( textures.size() ) );
                    }
                    else
                    {
                        material.textureIndex = textureIt->second;
                    }
                }

                int materialIndex = int( materials.size() );
                const auto materialIt = materials.find( material );
                if ( materialIt == materials.end() )
                {
                    materials.insert( { material, materialIndex } );
                }
                else
                {
                    materialIndex = materialIt->second;
                }

                const auto mesh = curObjectMesh->mesh();
                const auto points = mesh->points;
                const auto triangles = mesh->topology.getAllTriVerts();

                const size_t oldBufferEnd = buffer.size();
                const size_t verticesDataSize = points.size() * sizeof( Vector3f );
                const size_t trianglesDataSize = 3 * triangles.size() * sizeof( uint32_t );
                const size_t uvCoordsDataSize = ( material.textureIndex >= 0 ) ? points.size() * sizeof( Vector2f ) : 0;

                model.bufferViews.emplace_back();
                model.bufferViews.back().buffer = 0;
                model.bufferViews.back().byteOffset = oldBufferEnd;
                model.bufferViews.back().byteLength = verticesDataSize + trianglesDataSize + uvCoordsDataSize;

                buffer.resize( oldBufferEnd + model.bufferViews.back().byteLength );

                curNode.mesh = int( model.meshes.size() );
                model.meshes.emplace_back();
                auto& gltfMesh = model.meshes.back();
                model.meshes.back().primitives.emplace_back();
                auto& gltfPrimitive = gltfMesh.primitives.back();
                gltfPrimitive.mode = TINYGLTF_MODE_TRIANGLES;
                gltfPrimitive.material = materialIndex;

                gltfPrimitive.attributes.insert( { "POSITION", int( model.accessors.size() ) } );
                model.accessors.emplace_back();
                model.accessors.back().bufferView = int( model.bufferViews.size() - 1 );
                model.accessors.back().componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
                model.accessors.back().type = TINYGLTF_TYPE_VEC3;
                model.accessors.back().byteOffset = 0;
                model.accessors.back().count = points.size();
                std::copy( ( uint8_t* )points.data(), ( uint8_t* )( points.data() + points.size() ), &buffer[oldBufferEnd] );

                gltfPrimitive.indices = int( model.accessors.size() );
                model.accessors.emplace_back();
                model.accessors.back().bufferView = int( model.bufferViews.size() - 1 );
                model.accessors.back().componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
                model.accessors.back().type = TINYGLTF_TYPE_SCALAR;
                model.accessors.back().byteOffset = verticesDataSize;
                model.accessors.back().count = 3 * triangles.size();
                std::copy( ( uint8_t* )triangles.data(), ( uint8_t* )( triangles.data() + triangles.size() ), &buffer[oldBufferEnd + verticesDataSize] );

                if ( material.textureIndex >= 0 )
                {
                    gltfPrimitive.attributes.insert( { "TEXCOORD_0", int( model.accessors.size() ) } );
                    model.accessors.emplace_back();
                    model.accessors.back().bufferView = int( model.bufferViews.size() - 1 );
                    model.accessors.back().componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
                    model.accessors.back().type = TINYGLTF_TYPE_VEC2;
                    model.accessors.back().byteOffset = verticesDataSize + trianglesDataSize;
                    model.accessors.back().count = points.size();
                    const auto& uvCoords = curObjectMesh->getUVCoords();
                    std::copy( ( uint8_t* )uvCoords.data(), ( uint8_t* )( uvCoords.data() + uvCoords.size() ), &buffer[oldBufferEnd + verticesDataSize + trianglesDataSize] );
                }
            }

            for ( auto child : curObj->children() )
            {
                if ( child->isAncillary() )
                    continue;

                objectStack.push( child );
                indexStack.push( ++lastIndex );
                curNode.children.push_back( int( indexStack.top() ) );
            }
        }

        if ( callback && !callback( 0.1f + 0.7f * childIndex / root.children().size() ) )
            return unexpected( "Operation was cancelled" );
    }

    model.materials.resize( materials.size() );
    for ( const auto& materialIt : materials )
    {
        tinygltf::TextureInfo textureInfo;
        textureInfo.index = materialIt.first.textureIndex;
        textureInfo.texCoord = 0;
        auto& pbr = model.materials[materialIt.second].pbrMetallicRoughness;
        pbr.baseColorTexture = textureInfo;

        const auto& color = materialIt.first.baseColor;
        pbr.baseColorFactor = { color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a / 255.0f };
    }

    model.textures.resize( textures.size() );
    model.images.resize( textures.size() );
    model.samplers.resize( 1 );

    const auto prefix = utf8string( file.stem().u8string() );

    for ( const auto& textureIt : textures )
    {
        auto& image = model.images[textureIt.second];
        image.image.resize( textureIt.first.pixels.size() * sizeof( Color ) );
        std::copy( ( uint8_t* )textureIt.first.pixels.data(), ( uint8_t* )( textureIt.first.pixels.data() + textureIt.first.pixels.size() ), image.image.data() );

        image.bits = 8;
        image.component = 4;
        image.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
        image.uri = prefix + std::string( "_texture" ) + std::to_string( textureIt.second ) + std::string( ".png" );
        image.width = textureIt.first.resolution.x;
        image.height = textureIt.first.resolution.y;

        model.textures[textureIt.second].source = textureIt.second;
        model.textures[textureIt.second].sampler = 0;
    }

    if ( callback && !callback( 0.9f ) )
        return unexpected( "Operation was cancelled" );

    tinygltf::TinyGLTF writer;
    tinygltf::FsCallbacks fsCallbacks{ .FileExists = tinygltf::FileExists, .ExpandFilePath = tinygltf::ExpandFilePath, .ReadWholeFile = tinygltf::ReadWholeFile, .WriteWholeFile = tinygltf::WriteWholeFile };
    writer.SetImageWriter( tinygltf::WriteImageData, &fsCallbacks );

    const bool isBinary = file.extension() == u8".glb";

    if ( !writer.WriteGltfSceneToFile( &model, utf8string( file.u8string() ), isBinary, isBinary, true, isBinary ) )
        return unexpected( "File writing error" );

    if ( callback && !callback( 1.0f ) )
        return unexpected( "Operation was cancelled" );

    return {};
}

Expected<LoadedObject> loadObjectTreeFromGltf( const std::filesystem::path& file, const ProgressCallback& callback )
{
    return deserializeObjectTreeFromGltf( file, callback ).and_then(
        []( ObjectPtr && obj ) -> Expected<LoadedObject> { return LoadedObject{ .obj = std::move( obj ) }; } );
}

MR_ADD_SCENE_LOADER( IOFilter( "GL Transmission Format (.gltf,.glb)", "*.gltf;*.glb" ), loadObjectTreeFromGltf )

MR_ADD_SCENE_SAVER( IOFilter( "glTF JSON scene (.gltf)", "*.gltf" ), serializeObjectTreeToGltf )
MR_ADD_SCENE_SAVER( IOFilter( "glTF binary scene (.glb)", "*.glb" ), serializeObjectTreeToGltf )

} // namespace MR
#endif
