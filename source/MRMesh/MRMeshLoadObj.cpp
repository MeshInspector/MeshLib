#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRMatrix3.h"
#include "MRPch/MRTBB.h"
#include "MRQuaternion.h"
#include "MRObjectMesh.h"
#pragma warning( push )
#pragma warning( disable : 4062 4866 )
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#pragma warning( pop)


#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

#include <stack>

namespace
{
    using namespace MR;

    tl::expected<void, std::string> parseObjVertex( const std::string_view& str, Vector3f& v )
    {
        using namespace boost::spirit::x3;

        int i = 0;
        auto coord = [&] ( auto& ctx ) { v[i++] = _attr( ctx ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> float_[coord] >> float_[coord] >> float_[coord] ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse vertex in OBJ-file" );

        return {};
    }

    struct ObjFace
    {
        std::vector<int> vertices;
        std::vector<int> textures;
        std::vector<int> normals;
    };

    tl::expected<void, std::string> parseObjFace( const std::string_view& str, ObjFace& f )
    {
        using namespace boost::spirit::x3;

        auto v = [&] ( auto& ctx ) { f.vertices.emplace_back( _attr( ctx ) ); };
        auto vt = [&] ( auto& ctx ) { f.textures.emplace_back( _attr( ctx ) ); };
        auto vn = [&] ( auto& ctx ) { f.normals.emplace_back( _attr( ctx ) ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            // NOTE: actions are not being reverted after backtracking
            // https://github.com/boostorg/spirit/issues/378
            ( 'f' >> *( int_[v] >> -( '/' >> ( ( int_[vt] >> -( '/' >> int_[vn] ) ) | ( '/' >> int_[vn] ) ) ) ) ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse face in OBJ-file" );

        if ( f.vertices.empty() )
            return tl::make_unexpected( "Invalid face vertex count in OBJ-file" );
        if ( !f.textures.empty() && f.textures.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face texture count in OBJ-file" );
        if ( !f.normals.empty() && f.normals.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face normal count in OBJ-file" );
        return {};
    }
}

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Buffer<char> data( streamSize );
    // important on Windows: in stream must be open in binary mode, otherwise next will fail
    in.read( data.data(), (ptrdiff_t)data.size() );
    if ( !in )
        return tl::make_unexpected( std::string( "OBJ-format read error" ) );

    if ( callback && !callback( 0.25f ) )
        return tl::make_unexpected( "Loading canceled" );
    // TODO: redefine callback

    return fromSceneObjFile( data.data(), data.size(), combineAllObjects, callback );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    Triangulation t;

    auto finishObject = [&]() 
    {
        MR_NAMED_TIMER( "finish object" )
        if ( !t.empty() )
        {
            res.emplace_back();
            res.back().name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : t )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            for ( auto & vs : t )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            res.back().mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), t );
            t.clear();
        }
        currentObjName.clear();
    };

    Timer timer( "split by lines" );
    std::vector<size_t> newlines{ 0 };
    {
        constexpr size_t blockSize = 4096;
        const auto blockCount = (size_t)std::ceil( (float)size / blockSize );
        constexpr size_t maxGroupCount = 256;
        const auto blocksPerGroup = (size_t)std::ceil( (float)blockCount / maxGroupCount );
        const auto groupSize = blockSize * blocksPerGroup;
        const auto groupCount = (size_t)std::ceil( (float)size / groupSize );
        assert( groupCount <= maxGroupCount );
        assert( groupSize * groupCount >= size );
        assert( groupSize * ( groupCount - 1 ) < size );

        std::vector<std::vector<size_t>> groups( groupCount );
        tbb::task_group taskGroup;
        for ( size_t gi = 0; gi < groupCount; gi++ )
        {
            taskGroup.run( [&, i = gi]
            {
                std::vector<size_t> group;
                const auto begin = i * groupSize;
                const auto end = std::min( ( i + 1 ) * groupSize, size );
                for ( auto ci = begin; ci < end; ci++ )
                    if ( data[ci] == '\n' )
                        group.emplace_back( ci + 1 );
                groups[i] = std::move( group );
            } );
        }
        taskGroup.wait();

        size_t sum = newlines.size();
        std::vector<size_t> groupOffsets;
        for ( const auto& group : groups )
        {
            groupOffsets.emplace_back( sum );
            sum += group.size();
        }
        newlines.resize( sum );

        for ( size_t gi = 0; gi < groupCount; gi++ )
        {
            taskGroup.run( [&, i = gi]
            {
                const auto& group = groups[i];
                const auto offset = groupOffsets[i];
                for ( auto li = 0; li < group.size(); li++ )
                    newlines[offset + li] = group[li];
            } );
        }
        taskGroup.wait();
    }
    // add finish line
    if ( newlines.back() != size )
        newlines.emplace_back( size );
    const auto lineCount = newlines.size() - 1;

    if ( callback && !callback( 0.40f ) )
        return tl::make_unexpected( "Loading canceled" );

    timer.restart( "group element lines" );
    enum class Element
    {
        Unknown,
        Vertex,
        Face,
        Object,
    };
    struct ElementGroup
    {
        Element element{ Element::Unknown };
        size_t begin{ 0 };
        size_t end{ 0 };
    };
    std::vector<ElementGroup> groups{ { Element::Unknown, 0, 0 } }; // emplace stub initial group
    for ( size_t li = 0; li < lineCount; ++li )
    {
        auto* line = data + newlines[li];

        Element element = Element::Unknown;
        if ( line[0] == 'v' && line[1] != 'n' /*normals*/ && line[1] != 't' /*texture coordinates*/ )
        {
            element = Element::Vertex;
        }
        else if ( line[0] == 'f' )
        {
            element = Element::Face;
        }
        else if ( line[0] == 'o' )
        {
            element = Element::Object;
        }
        // TODO: multi-line elements

        if ( element != groups.back().element )
        {
            groups.back().end = li;
            groups.push_back( { element, li, 0 } );
        }
    }
    groups.back().end = lineCount;

    if ( callback && !callback( 0.50f ) )
        return tl::make_unexpected( "Loading canceled" );

    auto parseVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        const auto offset = points.size();
        points.resize( points.size() + ( end - begin ) );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3f v;
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjVertex( line, v );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                points[offset + ( li - begin )] = v;
            }
        }, ctx );
    };

    auto parseFaces = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        tbb::task_group_context ctx;
        tbb::enumerable_thread_specific<Triangulation> trisPerThread;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& tris = trisPerThread.local();

            ObjFace f;
            // usually a face has 3 or 4 vertices
            for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                elements->reserve( 4 );

            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                    elements->clear();

                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjFace( line, f );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }

                auto& vs = f.vertices;
                for ( auto& v : vs )
                {
                    if ( v < 0 )
                        v += (int)points.size() + 1;

                    if ( v <= 0 )
                    {
                        if ( ctx.cancel_group_execution() )
                            parseError = "Too negative vertex ID in OBJ-file";
                        return;
                    }
                }
                if ( vs.size() < 3 )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = "Face with less than 3 vertices in OBJ-file";
                    return;
                }

                // TODO: make smarter triangulation based on point coordinates
                for ( int j = 1; j + 1 < vs.size(); ++j )
                    tris.push_back( { VertId( vs[0]-1 ), VertId( vs[j]-1 ), VertId( vs[j+1]-1 ) } );
            }
        }, ctx );

        if ( !parseError.empty() )
            return;

        size_t trisSize = 0;
        for ( auto& tris : trisPerThread )
            trisSize += tris.size();
        t.reserve( t.size() + trisSize );
        for ( auto& tris : trisPerThread )
            t.vec_.insert( t.vec_.end(), tris.vec_.begin(), tris.vec_.end() );
    };

    auto parseObject = [&] ( size_t, size_t end, std::string& )
    {
        if ( combineAllObjects )
            return;

        // finish previous object
        finishObject();

        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        currentObjName = line.substr( 1, std::string_view::npos );
        boost::trim( currentObjName );
    };

    timer.restart( "parse groups" );
    for ( const auto& group : groups )
    {
        std::string parseError;
        switch ( group.element )
        {
        case Element::Unknown:
            break;
        case Element::Vertex:
            parseVertices( group.begin, group.end, parseError );
            break;
        case Element::Face:
            parseFaces( group.begin, group.end, parseError );
            break;
        case Element::Object:
            parseObject( group.begin, group.end, parseError );
            break;
        }
        if ( !parseError.empty() )
            return tl::make_unexpected( parseError );

        if ( callback && !callback( 0.50f + 0.50f * ( (float)group.end / (float)lineCount ) ) )
            return tl::make_unexpected( "Loading canceled" );
    }

    finishObject();
    return res;
}

tl::expected<std::vector<std::shared_ptr<Object>>, std::string> fromSceneGltfFile( const std::filesystem::path& file, bool, ProgressCallback callback)
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

    std::vector<MeshTexture> meshTextures;
    for ( auto& image : model.images )
    {
        meshTextures.emplace_back();
        auto& meshTexture = meshTextures.back();
        meshTexture.resolution = { image.width, image.height };
        meshTexture.pixels.resize( image.width * image.height );

        memcpy( meshTexture.pixels.data(), image.image.data(), image.width * image.height * sizeof( Color ) );
    }

    std::vector<Color> materialColors;
    std::vector<MeshTexture*> materialTextures;

    for ( auto& material : model.materials )
    {
        auto colorIt = material.values.find( "baseColorFactor" );
        if ( colorIt != material.values.end() )
        {
            const auto& comps = colorIt->second.number_array;
            materialColors.push_back( Color( float( comps[0] ), float( comps[1] ), float( comps[2] ), float ( comps[3]  ) ) );
        }
        else
        {
            materialColors.push_back( Color::white() );
        }

        const int textureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
        if ( textureIndex >= 0 && model.textures[textureIndex].source >= 0 )
        {
            materialTextures.push_back( &meshTextures[model.textures[textureIndex].source] );
        }
        else
        {
            materialTextures.push_back( nullptr );
        }
    }

    std::vector<std::shared_ptr<Mesh>> meshes;
    std::vector< Vector<Color, VertId> > vertColorMaps;
    std::vector< Vector<UVCoord, VertId> > uvCoordsVec;
    std::vector<int> meshMaterialIndices( model.meshes.size(), -1 );

    meshes.reserve( model.meshes.size() );
    vertColorMaps.reserve( model.meshes.size() );

    for ( size_t i = 0; i < model.meshes.size(); ++i )
    {
        const auto mesh = model.meshes[i];
        VertCoords vertexCoordinates;
        Triangulation t;
        vertColorMaps.emplace_back();
        uvCoordsVec.emplace_back();

        auto& vertColorMap = vertColorMaps.back();
        auto& uvCoords = uvCoordsVec.back();

        for ( const auto& primitive : mesh.primitives )
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
            
            
            if ( primitive.material < 0 )
            {
                vertColorMap.resize( vertColorMap.size() + accessor.count );
                memset( vertColorMap.data() + vertColorMap.size() - accessor.count, 0xFF, accessor.count * sizeof( Color ) );
            }
            else
            {
                if ( meshMaterialIndices[i] < 0 )
                    meshMaterialIndices[i] = primitive.material;

                vertColorMap.reserve( vertColorMap.size() + accessor.count );

                for ( int j = 0; j < accessor.count; ++j )
                {
                    vertColorMap.push_back( materialColors[primitive.material] );
                }
            }

            if ( primitive.indices < 0 )
            {
                t.reserve( t.size() + accessor.count / 3 );
                for ( int j = 0; j < accessor.count / 3; ++j )
                    t.push_back( { start + 3 * j, start + 3 * j + 1, start + 3 * j + 2 } );

                continue;
            }

            accessor = model.accessors[primitive.indices];
            bufferView = model.bufferViews[accessor.bufferView];
            buffer = model.buffers[bufferView.buffer];

            if ( accessor.componentType < TINYGLTF_COMPONENT_TYPE_BYTE || accessor.componentType > TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT || accessor.type != TINYGLTF_TYPE_SCALAR )
                return tl::make_unexpected( "Not implemented triangulation component type" );

            const auto fillTriangles = [start]<typename ComponentType>( Triangulation & t, size_t triangleCount, ComponentType * pBuffer )
            {
                for ( size_t k = 0; k < triangleCount; ++k )
                {
                    t.push_back( ThreeVertIds{ start + VertId( int(*pBuffer) ), start + VertId( int (*( pBuffer + 1 ) ) ), start + VertId( int ( *( pBuffer + 2 ) ) ) } );
                    pBuffer += 3;
                }
            };

            auto pBuffer = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
            const size_t triangleCount = accessor.count / 3;

            switch ( accessor.componentType )
            {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
                fillTriangles( t, triangleCount, (int8_t*)pBuffer );
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
                return tl::make_unexpected( "Not implemented triangulation component type" );
            };

            uvCoords.resize( uvCoords.size() + vertexCoordinates.size() );

            posAttrib = primitive.attributes.find( "TEXCOORD_0" );
            if ( posAttrib == primitive.attributes.end() )
            {
                continue;
            }

            accessor = model.accessors[posAttrib->second];
            bufferView = model.bufferViews[accessor.bufferView];
            buffer = model.buffers[bufferView.buffer];

            //assert( accessor.count == vertexCoordinates.size() );

            if ( ( accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT && accessor.componentType != TINYGLTF_COMPONENT_TYPE_DOUBLE ) || accessor.type != TINYGLTF_TYPE_VEC2 )
                return tl::make_unexpected( "Not implemented texcoord component type" );

            if ( accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT )
            {
                memcpy( uvCoords.data() + uvCoords.size() - accessor.count, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof( Vector2f ) );
            }
        }

        std::vector<MeshBuilder::VertDuplication> dups;
        meshes.push_back( std::make_shared<Mesh>( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinates, t, &dups ) ) );
        if ( !dups.empty() )
        {
            vertColorMap.resize( vertColorMap.size() + dups.size() );
            uvCoords.resize( uvCoords.size() + dups.size() );
            for ( const auto& dup : dups )
            {
                vertColorMap[dup.dupVert] = vertColorMap[dup.srcVert];
                uvCoords[dup.dupVert] = uvCoords[dup.srcVert];
            }
        }

        if ( !callback(0.8f * ( i + 1 ) / float( model.meshes.size() ) ) )
            return tl::make_unexpected( "Operation was cancelled" );
    }

    std::vector<std::shared_ptr<Object>> scene;
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

            int curNodeIndex = nodeStack.top();
            const auto& node = model.nodes[curNodeIndex];
            nodeStack.pop();
            auto curObject = objectStack.top();
            objectStack.pop();

            AffineXf3f xf;
            if ( node.matrix.size() == 16 )
            {
                xf = AffineXf3f(
                    Matrix3f{ { float( node.matrix[0] ), float( node.matrix[4] ), float( node.matrix[8] ) },
                              { float( node.matrix[1] ), float( node.matrix[5] ), float( node.matrix[9] ) },
                              { float( node.matrix[2] ), float( node.matrix[6] ), float( node.matrix[10] ) } },
                     Vector3f{ float( node.matrix[12] ), float( node.matrix[13] ), float( node.matrix[14] ) } );
            }
            else if ( node.translation.size() != 0 || node.rotation.size() != 0 || node.scale.size() != 0 )
            {
                if ( node.translation.size() == 3 )
                {
                   xf = xf * AffineXf3f::translation( { float( node.translation[0] ), float( node.translation[1] ), float( node.translation[2] ) } );
                }

                if ( node.rotation.size() == 4 )
                {
                    const Quaternion<float> q( float( node.rotation[3] ), float( node.rotation[0] ), float( node.rotation[1] ), float( node.rotation[2] ) );
                    xf = xf * AffineXf3f( Matrix3f( q ), {} );
                }

                if ( node.scale.size() == 3 )
                {
                    xf = xf * AffineXf3f( Matrix3f::scale( { float( node.scale[0] ), float( node.scale[1] ), float( node.scale[2] ) } ), {} );
                }
            }

            curObject->setXf( xf );
            curObject->setName( node.name );
            if ( node.mesh >= 0 )
            {
                auto objectMesh = std::static_pointer_cast< ObjectMesh >( curObject );
                objectMesh->setMesh( meshes[node.mesh] );               

                if ( materialTextures[meshMaterialIndices[node.mesh]] )
                {
                    objectMesh->setUVCoords( uvCoordsVec[node.mesh] );
                    objectMesh->setTexture( *materialTextures[meshMaterialIndices[node.mesh]] );                    
                    objectMesh->setVisualizeProperty( true, VisualizeMaskType::Texture, ViewportMask::all() );
                }
                else
                {
                    objectMesh->setColoringType( ColoringType::VertsColorMap );
                    objectMesh->setVertsColorMap( vertColorMaps[node.mesh] );
                    
                }
                //objectMesh->setTexture( model.meshes[node.mesh])
                if ( node.name.empty() )
                    curObject->setName( model.meshes[node.mesh].name );
            }
            else
            {
                curObject->setAncillary( true );
            }

            if ( curObject->name().empty() )
            {
                curObject->setName ( node.mesh >= 0 ? std::string( "Mesh " ) + std::to_string( meshCounter++ ) :
                    std::string( "Group " ) + std::to_string( groupCounter++ ) );
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

        scene.push_back( rootObject );
    }

    return scene;
}

} //namespace MeshLoad

} //namespace MR
