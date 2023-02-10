#include "MRLoadSceneFromGltf.h"
#include "MRVector.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRObjectMesh.h"
#include "MRMeshTexture.h"
#include "MRQuaternion.h"
#include "MRStringConvert.h"

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
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
#endif

#include<stack>

namespace
{
using namespace MR;

//holds together mesh itself, vertex colors, texture coordinates and index of material
struct MeshData
{
    std::shared_ptr<Mesh> mesh;
    Vector<Color, VertId> vertsColorMap;
    Vector<UVCoord, VertId> uvCoords;
    int materialIndex = -1;
};

//holds together base color and index of texture
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
    std::copy( &buffer.data[accessor.byteOffset + bufferView.byteOffset], &buffer.data[accessor.byteOffset + bufferView.byteOffset + accessor.count * sizeof( Vector3f )], ( uint8_t* )&vertexCoordinates[VertId( start )] );

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

    for ( size_t meshId = 0; meshId < model.meshes.size(); ++meshId )
    {
        if ( callback && !callback( 0.8f * ( meshId + 1 ) / float( model.meshes.size() ) ) )
            return tl::make_unexpected( "Operation was cancelled" );

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
                return tl::make_unexpected( vertexCount.error() );

            fillVertsColorMap( meshData.vertsColorMap, *vertexCount, materials, primitive.material );

            if ( auto error = readUVCoords( meshData.uvCoords, *vertexCount, model, primitive ); !error.empty() )
                return tl::make_unexpected( error );

            if ( auto error = readTriangulation( t, model, primitive, oldVertexCount, *vertexCount ); !error.empty() )
                return tl::make_unexpected( error );

            if ( meshData.materialIndex < 0 )
                meshData.materialIndex = primitive.material;

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
            meshData.vertsColorMap.resize( meshData.vertsColorMap.size() + dups.size() );

        meshData.uvCoords.resize( meshData.uvCoords.size() + dups.size() );
        for ( const auto& dup : dups )
        {
            if ( !areMaterialsSame )
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
            if ( callback && !callback( 0.8f + 0.2f * ( ++counter ) / float( model.nodes.size() ) ) )
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
                else if ( !meshData.vertsColorMap.empty() )
                {
                    objectMesh->setColoringType( ColoringType::VertsColorMap );
                    objectMesh->setVertsColorMap( meshData.vertsColorMap );
                }
                else
                {
                    objectMesh->setFrontColor( materials[meshData.materialIndex].baseColor, false );
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
}


