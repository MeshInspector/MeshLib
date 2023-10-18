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

// Todo for full compatibility:
// Parse Open Packaging Conventions structure (currently assumed most common layout)
// Determine namespace alias (currently assumed as 'p', e.g. "p:path")

namespace MR
{

namespace MeshLoad
{

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )

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

// Load mesh from <mesh> element
static Expected<Mesh, std::string> loadMesh( const tinyxml2::XMLElement* meshNode, ProgressCallback callback,
    int* duplicatedVertexCountAccum, int* skippedFaceCountAccum )
{
    auto verticesNode = meshNode->FirstChildElement( "vertices" );
    if ( !verticesNode )
        return unexpected( std::string( "3DF model 'vertices' node not found" ) );

    VertCoords vertexCoordinates;
    for ( auto vertexNode = verticesNode->FirstChildElement( "vertex" ); vertexNode;
          vertexNode = vertexNode->NextSiblingElement( "vertex" ) )
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
    if ( duplicatedVertexCountAccum )
        dupsPtr = &dups;
    MeshBuilder::BuildSettings buildSettings;
    if ( skippedFaceCountAccum )
    {
        skippedFaces = FaceBitSet( tris.size() );
        skippedFaces.set();
        buildSettings.region = &skippedFaces;
    }
    MR::Mesh mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris, dupsPtr, buildSettings );
    if ( duplicatedVertexCountAccum )
        *duplicatedVertexCountAccum += int( dups.size() );
    if ( skippedFaceCountAccum )
        *skippedFaceCountAccum += int( skippedFaces.count() );

    if ( !reportProgress( callback, 0.75f ) )
        return unexpected( std::string( "Loading canceled" ) );

    return mesh;
}

namespace
{

class ThreeMFLoader
{
    // Documents index
    std::map<std::filesystem::path, std::unique_ptr<tinyxml2::XMLDocument>> m_documents;
    std::filesystem::path m_rootPath;

    // Object tree - each node is either a mesh or compound object
    struct Node
    {
        AffineXf3f xf;
        const tinyxml2::XMLElement* meshNode = nullptr;
        std::vector<Node> childNodes;
    };
    Node m_rootNode;
    int m_meshesCount = 0; // For progress indication

    // Mesh cache
    std::map<const tinyxml2::XMLElement*, Mesh> m_meshCache;

    // Load and parse all XML .model files
    VoidOrErrStr loadXmls( const std::vector<std::filesystem::path>& files );

    // Load object tree from loaded XML files
    VoidOrErrStr loadTree();
    // Create object node from either <build> element (default) or from <object> with given id in <resources>
    Expected<Node, std::string> loadNodeFromDocument( const tinyxml2::XMLDocument& doc, std::string id = {} );
    // Create node containing mesh object (<object>) or multi-component object form <components> or <build> element
    Expected<Node, std::string> createNode( const tinyxml2::XMLElement* element, const tinyxml2::XMLDocument& doc );

    // Load meshes (fill m_meshCache)
    VoidOrErrStr loadMeshes( const Node& startNode );

    // Assemble mesh from tree and meshes
    Expected<Mesh, std::string> createMeshFromNode( const Node& node, ProgressCallback subCallback );

public:
    ProgressCallback callback = {};
    int* duplicatedVertexCount = nullptr, * skippedFaceCount = nullptr;

    // Load from .model files
    Expected<Mesh, std::string> load( const std::vector<std::filesystem::path>& files, std::filesystem::path root );
};

VoidOrErrStr ThreeMFLoader::loadXmls( const std::vector<std::filesystem::path>& files )
{
    for ( std::filesystem::path file : files )
    {
        std::ifstream in( file, std::ifstream::binary );
        if ( !in )
            return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
        // Read file contents to char vector
        in.seekg( 0, std::ios_base::end );
        size_t size = in.tellg();
        in.seekg( 0 );
        std::vector<char> docStr( size + 1 );
        in.read( docStr.data(), size );
        if ( in.fail() || in.bad() )
            return unexpected( std::string( "3DF model file read error" ) + utf8string( file ) );
        // Parse XML
        auto doc = std::make_unique<tinyxml2::XMLDocument>();
        if ( tinyxml2::XML_SUCCESS != doc->Parse( docStr.data(), docStr.size() ) ||
             doc->FirstChildElement() == nullptr )
            return unexpected( std::string( "3DF model file parse error" ) + utf8string( file ) );
        // Store parsed XML
        m_documents.emplace( file.lexically_normal(), std::move( doc ) );
    }
    return {};
}

VoidOrErrStr ThreeMFLoader::loadTree()
{
    for ( auto& [file, doc] : m_documents )
    {
        // Start from <build> elements of models
        auto res = loadNodeFromDocument( *doc );
        if ( !res )
            return unexpected( res.error() );
        if ( m_documents.size() == 1 )
            m_rootNode = std::move( *res );
        else
            m_rootNode.childNodes.emplace_back( *res );
    }
    return {};
}

// Load an object from either <build> element (default) or from <object> with given id in <resources>
Expected<ThreeMFLoader::Node, std::string> ThreeMFLoader::loadNodeFromDocument(
    const tinyxml2::XMLDocument& doc, std::string id )
{
    auto rootNode = doc.FirstChildElement();
    if ( std::string( rootNode->Name() ) != "model" )
        return unexpected( std::string( "3DF model root node is not 'model' but '" ) + rootNode->Name() + "'" );

    if ( id.empty() )
        // Load from <build> element
        return createNode( rootNode, doc );

    auto resourcesNode = rootNode->FirstChildElement( "resources" );
    if ( !resourcesNode )
        return unexpected( "3DF model 'resources' node not found" );
    for ( auto objectNode = resourcesNode->FirstChildElement( "object" ); objectNode;
        objectNode = objectNode->NextSiblingElement( "object" ) )
    {
        if ( objectNode->Attribute( "id", id.c_str() ) )
            // Load from <mesh> or <components> element
            return createNode( objectNode, doc );
    }
    return unexpected( "3DF object '" + id + "' not found" );
}

Expected<ThreeMFLoader::Node, std::string> ThreeMFLoader::createNode(
    const tinyxml2::XMLElement* parentElement, const tinyxml2::XMLDocument& doc )
{
    ThreeMFLoader::Node node;
    const tinyxml2::XMLElement* element;
    const char* childTag = nullptr;
    // Find child element
    for ( element = parentElement->FirstChildElement(); element; element = element->NextSiblingElement( childTag ) )
    {
        std::string name = element->Name();
        if ( name == "mesh" )
            break; // Leave childTag == nullptr
        if ( name == "components" )
            childTag = "component";
        if ( name == "build" )
            childTag = "item";
        if ( childTag )
            break;
    }
    if ( !element )
        return unexpected( "No object in '" + std::string( parentElement->Name() ) + "'" );
    // Mesh element
    if ( childTag == nullptr )
    {
        node.meshNode = element;
        m_meshesCount++;
        return node;
    }
    // Compound element
    for ( auto objectNode = element->FirstChildElement( childTag ); objectNode;
          objectNode = objectNode->NextSiblingElement( childTag ) )
    {
        auto idAttr = objectNode->FindAttribute( "objectid" );
        auto pathAttr = objectNode->FindAttribute( "p:path" );
        const tinyxml2::XMLDocument* docPtr = &doc;
        if ( pathAttr )
        {
            std::filesystem::path filePath = m_rootPath;
            filePath.concat( "/" ).concat( pathAttr->Value() ); // "path" value is e.g. "/3D/other.model"
            auto docIt = m_documents.find( filePath.lexically_normal() );
            if ( docIt == m_documents.end() )
                return unexpected( "Could not find '" + std::string( pathAttr->Value() ) + "'" );
            docPtr = docIt->second.get();
        }
        auto res = loadNodeFromDocument( *docPtr, idAttr ? idAttr->Value() : "" );
        if ( !res )
            return res;
        ThreeMFLoader::Node subNode = std::move( *res );
        if ( auto transformAttr = objectNode->FindAttribute( "transform" ) )
        {
            auto resXf = parseAffineXf( transformAttr->Value() );
            if ( !resXf )
                return unexpected( resXf.error() );
            subNode.xf = *resXf;
        }
        node.childNodes.emplace_back( subNode );
    }
    return node;
}

VoidOrErrStr ThreeMFLoader::loadMeshes( const Node& startNode )
{
    if ( duplicatedVertexCount )
        *duplicatedVertexCount = 0;
    if ( skippedFaceCount )
        *skippedFaceCount = 0;
    if ( startNode.meshNode )
    {
        if ( m_meshCache.contains( startNode.meshNode ) )
            return {};
        float itemsLoaded = float( m_meshCache.size() );
        auto res = loadMesh( startNode.meshNode,
            subprogress( callback, 0.2f + itemsLoaded / m_meshesCount * 0.7f,
                0.2f + ( itemsLoaded + 1 ) / m_meshesCount * 0.7f ),
            duplicatedVertexCount, skippedFaceCount );
        if ( !res )
            return unexpected( res.error() );
        m_meshCache.emplace( startNode.meshNode, *res );
    }
    else
    {
        for ( const Node& node : startNode.childNodes )
            loadMeshes( node );
    }
    return {};
}

Expected<Mesh, std::string> ThreeMFLoader::createMeshFromNode( const Node& node, ProgressCallback subCallback )
{
    Mesh mesh;
    if ( node.meshNode )
    {
        mesh = m_meshCache[node.meshNode];
    }
    else
    {
        float progress = 0.0f;
        for ( const Node& n : node.childNodes )
        {
            auto res = createMeshFromNode( n,
                subprogress( subCallback, progress, progress + 1.0f / node.childNodes.size() ) );
            if ( !res )
                return res;
            mesh.addPart( std::move( *res ) );
            progress += 1.0f / node.childNodes.size();
        }
    }
    if ( !reportProgress( subCallback, 0.5f ) )
        return unexpected( std::string( "Loading canceled" ) );
    if ( node.xf != AffineXf3f() )
        mesh.transform( node.xf );
    return mesh;
}

Expected<Mesh, std::string> ThreeMFLoader::load( const std::vector<std::filesystem::path>& files, std::filesystem::path root )
{
    m_rootPath = root.lexically_normal();

    auto res = loadXmls( files );
    if ( !res )
        return unexpected( res.error() );

    if ( !reportProgress( callback, 0.2f ) )
        return unexpected( std::string( "Loading canceled" ) );

    res = loadTree();
    if ( !res )
        return unexpected( res.error() );

    res = loadMeshes( m_rootNode );
    if ( !res )
        return unexpected( res.error() );

    if ( !reportProgress( callback, 0.9f ) )
        return unexpected( std::string( "Loading canceled" ) );

    return createMeshFromNode( m_rootNode, subprogress( callback, 0.9f, 1.0f ) );
}

} //namespace

static Expected<Mesh, std::string> doLoad(
    const std::vector<std::filesystem::path>& files, std::filesystem::path root,
    const MeshLoadSettings& settings /*= {}*/ )
{
    ThreeMFLoader loader;
    loader.callback = settings.callback;
    loader.duplicatedVertexCount = settings.duplicatedVertexCount;
    loader.skippedFaceCount = settings.skippedFaceCount;

    return loader.load( files, root );
}

Expected<Mesh, std::string> from3mf( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    if ( file.extension() == ".model" )
        return addFileNameInError(
            doLoad( { file }, file.parent_path(), settings ), file );

    return addFileNameInError( from3mf( in, settings ), file );
}

Expected<Mesh, std::string> from3mf( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    auto tmpFolder = UniqueTemporaryFolder( {} );

    auto resZip = decompressZip( in, tmpFolder );
    if ( !resZip )
        return unexpected( "ZIP container error: " + resZip.error() );

    if ( !reportProgress(settings.callback, 0.1f ) )
        return unexpected( std::string( "Loading canceled" ) );

    // Search for common locations (usually it is "3D/*.model")
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    // Errors are not expected here (except "3D" directory not existing), so don't handle them
    for ( auto const& dirEntry : Directory{ tmpFolder / "3D", ec } )
        if (dirEntry.path().extension() == ".model" )
            files.push_back( dirEntry.path() );
    if ( files.empty() ) // Handle unusual layouts
        for ( auto const& dirEntry : DirectoryRecursive{ tmpFolder, ec } )
            if ( dirEntry.path().extension() == ".model" )
                files.push_back( dirEntry.path() );
    if ( files.empty() )
        return unexpected( "Could not find .model" );

    return doLoad( files, tmpFolder, settings );
}

#endif

} //namespace MeshLoad

} //namespace MR
