#include "MRGltfSerializer.h"
#include "MRVector.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRObjectMesh.h"
#include "MRMeshTexture.h"
#include "MRQuaternion.h"
#include "MRStringConvert.h"
#include "MRSceneRoot.h"

#if (defined(__APPLE__) && defined(__clang__))
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#pragma warning( push )
#pragma warning( disable : 4062 4866 )
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#pragma warning( pop)

#if (defined(__APPLE__) && defined(__clang__))
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

    bool operator==( const Material& other ) const
    {
        return baseColor == other.baseColor && textureIndex == other.textureIndex;
    }
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
        {
            meshData.vertsColorMap.resize( meshData.vertsColorMap.size() + dups.size() );
            for ( const auto& dup : dups )
            {
                meshData.vertsColorMap[dup.dupVert] = meshData.vertsColorMap[dup.srcVert];
            }
        }

        if ( model.materials[meshData.materialIndex].pbrMetallicRoughness.baseColorTexture.index >= 0 )
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

tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback )
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
    {        
        return tl::make_unexpected( err );
    }

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

                if ( meshData.materialIndex >= 0 )
                {
                    const auto textureIndex = materials[meshData.materialIndex].textureIndex;
                    if ( textureIndex >=0 && model.textures[textureIndex].source >=0 )
                    {
                        objectMesh->setUVCoords( meshData.uvCoords );
                        objectMesh->setTexture( textures[model.textures[textureIndex].source] );
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

tl::expected<void, std::string> serializeObjectTreeToGltf( const Object& root, const std::filesystem::path& file, ProgressCallback callback )
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

    std::unordered_map<MeshTexture, int, decltype(textureHash), decltype(textureCompare)> textures( {}, textureHash, textureCompare);


    std::stack<std::shared_ptr<const Object>> objectStack;
    std::stack<size_t> indexStack;

    for ( auto rootChild : root.children() )
    {
        objectStack.push( rootChild );
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
                objectStack.push( child );
                indexStack.push( ++lastIndex );
                curNode.children.push_back( int( indexStack.top() ) );
            }
        }
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

    for ( const auto& textureIt : textures )
    {
        auto& image = model.images[textureIt.second];
        image.image.resize( textureIt.first.pixels.size() * sizeof( Color ) );
        std::copy( ( uint8_t* )textureIt.first.pixels.data(), ( uint8_t* )( textureIt.first.pixels.data() + textureIt.first.pixels.size() ), image.image.data() );

        image.bits = 8;
        image.component = 4;
        image.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
        image.uri = std::string( "texture" ) + std::to_string( textureIt.second ) + std::string( ".png" );
        image.width = textureIt.first.resolution.x;
        image.height = textureIt.first.resolution.y;

        model.textures[textureIt.second].source = textureIt.second;
        model.textures[textureIt.second].sampler = 0;
    }

    tinygltf::TinyGLTF writer;
    tinygltf::FsCallbacks fsCallbacks{ .FileExists = tinygltf::FileExists, .ExpandFilePath = tinygltf::ExpandFilePath, .ReadWholeFile = tinygltf::ReadWholeFile, .WriteWholeFile = tinygltf::WriteWholeFile };
    writer.SetImageWriter( tinygltf::WriteImageData, &fsCallbacks );

    const bool isBinary = file.extension() == u8".glb";

    if ( !writer.WriteGltfSceneToFile( &model, utf8string( file.u8string() ), isBinary, isBinary, true, isBinary ) )
        return tl::make_unexpected( "File writing error" );

    return {};
}
}


