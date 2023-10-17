#include "MRMeshLoad.h"
#include "MRMeshBuilder.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRStringConvert.h"
#include "MRSerializer.h"
#include "MRZip.h"
#include "MRDirectory.h"
#include "MRPch/MRSpdlog.h"

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
#include <tinyxml2.h>
#endif

namespace MR
{

namespace MeshLoad
{

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
Expected<Mesh, std::string> from3mfModel( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( from3mfModel( in, settings ), file );
}

// Parse affine matrix as 4 rows of 3 elements in a space-separated string
// E.g. translation "1.0000 0.0000 0.0000 0.0000 1.0000 0.0000 0.0000 0.0000 1.0000 0.1000 -4.0000 20.0000"
static Expected <AffineXf3f, std::string> parseAffineXf( std::string s )
{
    std::istringstream ss( s );
    float value;
    AffineXf3f xf;
    int row = 0, col = 0;
    while ( ss >> value )
    {
        if ( row < 3 )
            xf.A[row][col] = value;
        else if ( row == 3 )
            xf.b[col] = value;
        col++;
        if ( col == 3 )
            col = 0, row++;
    }
    if ( !( row == 4 && col == 0 ) )
        return unexpected( "Invalid matrix format" );
    return xf;
}

namespace
{

struct ThreeMFLoader
{
    // Nodes index
    std::map<std::string, const tinyxml2::XMLElement*> meshNodes, componentsNodes;

    // For progress indication
    int itemsCount = 0;
    int itemsLoaded = 0;
    ProgressCallback callback = {};

    // Mesh cache
    // Meshes and compound objects (containing other meshes/compounds) can go in the file in any order
    // So they are created recursively with the usage of the cache
    std::map<std::string, Mesh> resources;

    // Output information
    int *duplicatedVertexCount = nullptr, *skippedFaceCount = nullptr;
    std::string error;
};

ProgressCallback makeCallback( ThreeMFLoader &loader )
{
    int i = std::min( loader.itemsLoaded, loader.itemsCount - 1 );
    return subprogress( loader.callback, 0.1f + i / loader.itemsCount * 0.8f,
        0.1f + ( loader.itemsLoaded + 1 ) / loader.itemsCount * 0.9f );
}

// Load an object, using the resource cache
Expected<Mesh, std::string> loadObject( const char* id, ThreeMFLoader& loader );

// Load mesh from <mesh> element
Expected<Mesh, std::string> loadMesh( const tinyxml2::XMLElement* meshNode, ProgressCallback callback,
    ThreeMFLoader &loader )
{
    auto verticesNode = meshNode->FirstChildElement( "vertices" );
    if ( !verticesNode )
        return unexpected( std::string( "3DF model 'vertices' node not found" ) );

    VertCoords vertexCoordinates;
    for ( auto vertexNode = verticesNode->FirstChildElement( "vertex" ); vertexNode; vertexNode = vertexNode->NextSiblingElement( "vertex" ) )
    {
        Vector3f p;
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "x", &p.x ) )
            return unexpected( std::string( "3DF model vertex node does not have 'x' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "y", &p.y ) )
            return unexpected( std::string( "3DF model vertex node does not have 'y' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "z", &p.z ) )
            return unexpected( std::string( "3DF model vertex node does not have 'z' attribute" ) );
        vertexCoordinates.push_back( p );
    }

    if ( !reportProgress( callback, 0.25f ) )
        return unexpected( std::string( "Loading canceled" ) );

    auto trianglesNode = meshNode->FirstChildElement( "triangles" );
    if ( !trianglesNode )
        return unexpected( std::string( "3DF model 'triangles' node not found" ) );

    Triangulation tris;
    for ( auto triangleNode = trianglesNode->FirstChildElement( "triangle" ); triangleNode; triangleNode = triangleNode->NextSiblingElement( "triangle" ) )
    {
        int vs[3];
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v1", &vs[0] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v1' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v2", &vs[1] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v2' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v3", &vs[2] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v3' attribute" ) );
        tris.push_back( { VertId( vs[0] ), VertId( vs[1] ), VertId( vs[2] ) } );
    }

    if ( !reportProgress( callback, 0.5f ) )
        return unexpected( std::string( "Loading canceled" ) );

    FaceBitSet skippedFaces;
    std::vector<MeshBuilder::VertDuplication> dups;
    std::vector<MeshBuilder::VertDuplication>* dupsPtr = nullptr;
    if ( loader.duplicatedVertexCount )
        dupsPtr = &dups;
    MeshBuilder::BuildSettings buildSettings;
    if ( loader.skippedFaceCount )
    {
        skippedFaces = FaceBitSet( tris.size() );
        skippedFaces.set();
        buildSettings.region = &skippedFaces;
    }
    MR::Mesh mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris, dupsPtr, buildSettings );
    if ( loader.duplicatedVertexCount )
        *loader.duplicatedVertexCount += int( dups.size() );
    if ( loader.skippedFaceCount )
        *loader.skippedFaceCount += int( skippedFaces.count() );

    if ( !reportProgress( callback, 0.75f ) )
        return unexpected( std::string( "Loading canceled" ) );

    return mesh;
}

// Parse multi-component object form <components> or <build> element
Expected<Mesh, std::string> loadComponents(
    const tinyxml2::XMLElement* componentsNode, const char* childTagName, ProgressCallback callback, ThreeMFLoader &loader )
{
    Mesh resultMesh;
    for ( auto objectNode = componentsNode->FirstChildElement( childTagName ); objectNode; 
          objectNode = objectNode->NextSiblingElement( childTagName ) )
    {
        auto res = loadObject( objectNode->FindAttribute( "objectid" )->Value(), loader );
        if ( !res )
            return res;
        Mesh mesh = std::move( *res );
        // Transform mesh
        if ( auto transformNode = objectNode->FindAttribute( "transform" ) )
        {
            auto resXf = parseAffineXf( transformNode->Value() );
            if ( !resXf )
                return unexpected( resXf.error() );
            if ( *resXf != AffineXf3f() )
                mesh.transform( *resXf );
        }

        if ( !reportProgress( callback, 0.5f ) )
            return unexpected( std::string( "Loading canceled" ) );

        resultMesh.addPart( mesh );
    }
    return resultMesh;
}

// Load an object, using the resource cache
Expected<Mesh, std::string> loadObject( const char* id, ThreeMFLoader &loader )
{
    if ( loader.resources.contains( id ) )
        return loader.resources[id];
    Expected<Mesh, std::string> res = unexpected( "3DF object '" + utf8string( id ) + "' not found" );
    if ( loader.meshNodes.contains( id ) )
        res = loadMesh( loader.meshNodes[id], makeCallback( loader ), loader );
    else if ( loader.componentsNodes.contains( id ) )
        res = loadComponents( loader.componentsNodes[id], "component", makeCallback( loader ), loader );
    if ( !res )
        return res;
    loader.itemsLoaded++;
    loader.resources[id] = *res;
    return res;
}

// Index nodes and calculate basic parameters
VoidOrErrStr prepareLoader( const tinyxml2::XMLElement* rootNode, ThreeMFLoader &loader )
{
    auto resourcesNode = rootNode->FirstChildElement( "resources" );
    if ( !resourcesNode )
        return unexpected( "3DF model 'resources' node not found" );

    for ( auto objectNode = resourcesNode->FirstChildElement( "object" ); objectNode; objectNode = objectNode->NextSiblingElement( "object" ) )
    {
        if ( const char* attrId = objectNode->Attribute( "id" ) )
        {
            if ( auto meshNode = objectNode->FirstChildElement( "mesh" ) )
                loader.meshNodes[attrId] = meshNode;
            else if ( auto componentsNode = objectNode->FirstChildElement( "components" ) )
                loader.componentsNodes[attrId] = componentsNode;
        }
    }
    loader.itemsCount = int( loader.meshNodes.size() + loader.componentsNodes.size() ) + 1;
    return {};
}

} //namespace

Expected<Mesh, std::string> from3mfModel( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    // find size
    in.seekg( 0, std::ios_base::end );
    size_t size = in.tellg();
    in.seekg( 0 );
    // read to char vector
    std::vector<char> docStr( size + 1 );
    in.read( docStr.data(), size );
    if ( in.fail() || in.bad() )
        return unexpected( std::string( "3DF model file read error" ) );

    tinyxml2::XMLDocument doc;
    if ( tinyxml2::XML_SUCCESS != doc.Parse( docStr.data(), docStr.size() ) )
        return unexpected( std::string( "3DF model file parse error" ) );

    auto rootNode = doc.FirstChildElement();
    if ( !rootNode || std::string( rootNode->Name() ) != "model" )
        return unexpected( std::string( "3DF model root node is not 'model' but '" ) +
            std::string( rootNode ? rootNode->Name() : "" ) + "'");

    ThreeMFLoader loader;

    // Fill values and prepare elements references
    loader.callback = settings.callback;
    loader.duplicatedVertexCount = settings.duplicatedVertexCount;
    if ( loader.duplicatedVertexCount )
        *loader.duplicatedVertexCount = 0;
    loader.skippedFaceCount = settings.skippedFaceCount;
    if ( loader.skippedFaceCount )
        *loader.skippedFaceCount = 0;
    if ( !prepareLoader( rootNode, loader ) )
        return unexpected( loader.error );
    
    if ( !reportProgress( loader.callback, 0.1f ) )
        return unexpected( std::string( "Loading canceled" ) );

    // Load scene, starting from the <build> element
    auto buildNode = rootNode->FirstChildElement( "build" );
    if ( !buildNode )
        return unexpected( std::string( "3DF model 'build' node not found" ) );
    auto res = loadComponents( buildNode, "item", makeCallback(loader), loader );
    if (!res)
        return unexpected( res.error() );
    return *res;
}

Expected<Mesh, std::string> from3mf( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( from3mf( in, settings ), file );
}

Expected<Mesh, std::string> from3mf( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    auto tmpFolder = UniqueTemporaryFolder( {} );

    auto res = decompressZip( in, tmpFolder );
    if ( !res )
        return unexpected( "ZIP container error: " + res.error() );

    // Search for common locations (usually it is "3D/*.model")
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    // Errors are not expected here (except "3D" directory not existing), so don't handle them
    for ( auto const& dirEntry : Directory{ tmpFolder / "3D", ec } )
        files.push_back( dirEntry.path() );
    for ( auto const& dirEntry : DirectoryRecursive{ tmpFolder, ec } )
        files.push_back( dirEntry.path() );

    std::filesystem::path modelFilePath;
    for ( auto const& path: files )
        if ( path.extension() == ".model" )
        {
            modelFilePath = path;
            break;
        }
    if ( modelFilePath.empty() )
        return unexpected( "Could not find .model" );

    return from3mfModel( modelFilePath, settings );
}
#endif

} //namespace MeshLoad

} //namespace MR
