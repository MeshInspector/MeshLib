#include "MR3mf.h"
#ifndef MRIOEXTRAS_NO_3MF

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRZip.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRImageLoad.h"
#include "MRPch/MRFmt.h"

#include <tinyxml2.h>

#include <charconv>
#include <cmath>
#include <unordered_map>

namespace MR
{

static Expected <AffineXf3f> parseAffineXf( const std::string& s )
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

static Expected<Color> parseColor( const std::string& s )
{    
    if ( ( s.size() != 7 && s.size() != 9 ) || s[0] != '#' )
        return unexpected( "Invalid color format" );

    Color res;
    if ( std::from_chars( s.data() + 1, s.data() + 3, res.r, 16 ).ec != std::errc() )
        return unexpected( "Invalid color format" );

    if ( std::from_chars( s.data() + 3, s.data() + 5, res.g, 16 ).ec != std::errc() )
        return unexpected( "Invalid color format" );

    if ( std::from_chars( s.data() + 5, s.data() + 7, res.b, 16 ).ec != std::errc() )
        return unexpected( "Invalid color format" );

    if ( s.size() == 9 )
        if ( std::from_chars( s.data() + 7, s.data() + 9, res.a, 16 ).ec != std::errc() )
            return unexpected( "Invalid color format" );

    return res;
}

enum class NodeType
{
    Unknown,
    Model,
    Object,
    ColorGroup,
    Texture2d,
    Texture2dGroup,
    Build,
    BaseMaterials,
    Multiproperties
};

class ThreeMFLoader;

static const std::unordered_map<std::string, NodeType> nodeTypeMap =
{
    { "model", NodeType::Model },
    { "object", NodeType::Object },
    { "build", NodeType::Build },
    { "m:colorgroup", NodeType::ColorGroup },
    { "m:texture2d", NodeType::Texture2d },
    { "m:texture2dgroup", NodeType::Texture2dGroup },
    { "basematerials", NodeType::BaseMaterials },
    { "m:multiproperties", NodeType::Multiproperties }
};

static Expected<std::vector<int>> parseInts( const std::string& str )
{
    std::vector<int> res;
    size_t l = str.find_first_not_of( ' ' );
    while ( l != std::string::npos )
    {
        size_t r = str.find( ' ', l );
        if ( r == std::string::npos )
            r = str.size();

        int v = {};
        if ( std::from_chars( &str[l], &str[r], v ).ec != std::errc() )
            return unexpected( "Parsing of int vector failed" );
        res.push_back( v );

        if ( r == str.size() )
            break;

        l = str.find_first_not_of( ' ', r + 1 );
    }
    return res;
}

class ThreeMFLoader;
class Node
{
private:    

    Expected<void> loadColorGroup_( const tinyxml2::XMLElement* xmlNode );
    Expected<void> loadBaseMaterials_( const tinyxml2::XMLElement* xmlNode );
    Expected<void> loadObject_( const tinyxml2::XMLElement* xmlNode, ProgressCallback callback );    
    Expected<void> loadBuildData_( const tinyxml2::XMLElement* xmlNode );
    Expected<void> loadTexture2d_( const tinyxml2::XMLElement* xmlNode );
    Expected<void> loadTexture2dGroup_( const tinyxml2::XMLElement* xmlNode );
    Expected<void> loadMultiproperties_( const tinyxml2::XMLElement* xmlNode );
    
    Expected<Mesh> loadMesh_( const tinyxml2::XMLElement* meshNode, ProgressCallback callback );

    int id = -1;
    int pid = -1;
    int pindex = -1;
    int texId = -1;

    Node* pNode = nullptr;

    NodeType nodeType = NodeType::Unknown;
    std::vector<std::shared_ptr<Node>> children;
    std::string nodeName;
    std::string objName;

    Mesh mesh;
    std::vector<Color> colors;
    Color bgColor = Color::white();

    MeshTexture texture;
    std::vector<UVCoord> uvCoords;
    VertUVCoords vertUVCoords;
    VertColors vertColorMap;
    AffineXf3f xf;

    std::vector<int> pids;
    std::vector<std::vector<int>> pindices;
    const tinyxml2::XMLElement* node = nullptr;

public:
    
    inline static ThreeMFLoader* loader = nullptr;
      

    Node( const tinyxml2::XMLElement* xmlNode )
    : nodeName( xmlNode->Name() ),
    node( xmlNode )
    {
    }

    Expected<void> load();

    friend class ThreeMFLoader;
};

class ThreeMFLoader
{
    // Documents index
    std::vector<std::unique_ptr<tinyxml2::XMLDocument>> documents_;
    std::filesystem::path rootPath_;    
    // Object tree - each node is either a mesh or compound object

    std::unordered_map<int, Node*> idToNodeMap_;
    std::vector<Node*> objectNodes_;

    std::vector<std::shared_ptr<Node>> roots_;

    Expected<std::unique_ptr<tinyxml2::XMLDocument>> loadXml_( const std::filesystem::path& file );
    // Load and parse all XML .model files
    Expected<void> loadXmls_( const std::vector<std::filesystem::path>& files );

    // Load object tree from loaded XML files
    Expected<void> loadTree_( ProgressCallback callback );
    Expected<void> loadDocument_( std::unique_ptr<tinyxml2::XMLDocument>& doc, ProgressCallback callback );

    int duplicatedVertexCountAccum = 0;
    int skippedFaceCountAccum = 0;    
    
    ProgressCallback documentProgressCallback;
    ProgressCallback generalCallback;

    size_t objectCount_ = 0;
    size_t objectsLoaded_ = 0;
    size_t documentsLoaded_ = 0;

public:
    
    std::string* loadWarn = nullptr;
    bool failedToLoadColoring = false;

    Expected<std::shared_ptr<Object>> load( const std::vector<std::filesystem::path>& files, std::filesystem::path root, ProgressCallback callback );

    friend class Node;
};

Expected<std::unique_ptr<tinyxml2::XMLDocument>> ThreeMFLoader::loadXml_( const std::filesystem::path& file )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    //check if it is an XML file
    char header[6] = {};
    in.read( header, 5 );
    if ( in.fail() || in.bad() )
        return unexpected( std::string( "3DF model file read error" ) + utf8string( file ) );

    //if not just return nullptr, it is not an error
    if ( std::string( header, 5 ) != "<?xml" )
        return nullptr;

    in.seekg( 0 );
    // Read file contents to char buffer
    auto docBuf = readCharBuffer( in );
    if ( !docBuf )
        return unexpected( std::string( "3DF model file read error" ) + utf8string( file ) );
    // Parse XML
    auto doc = std::make_unique<tinyxml2::XMLDocument>();
    if ( tinyxml2::XML_SUCCESS != doc->Parse( docBuf->data(), docBuf->size() ) ||
         doc->FirstChildElement() == nullptr )
        return unexpected( std::string( "3DF model file parse error" ) + utf8string( file ) );

    return doc;
}

Expected<void> ThreeMFLoader::loadXmls_( const std::vector<std::filesystem::path>& files )
{
    for ( std::filesystem::path file : files )
    {
        auto docRes = loadXml_( file );
        if ( !docRes )
            return unexpected( docRes.error() );

        if ( *docRes == nullptr )
            continue;
        // Store parsed XML
        documents_.push_back( std::move( *docRes ) );
    }
    return {};
}

Expected<void> ThreeMFLoader::loadDocument_( std::unique_ptr<tinyxml2::XMLDocument>& doc, ProgressCallback callback )
{
    auto xmlNode = doc->FirstChildElement();
    if ( std::string( xmlNode->Name() ) != "model" ) //maybe another xml, just skip
        return {};

    objectCount_ = 0;
    documentProgressCallback = callback;

    auto resourcesNode = xmlNode->FirstChildElement( "resources" );
    if ( !resourcesNode )
        return unexpected( std::string( "3DF model <build> tag not found" ) );

    for ( auto itemNode = resourcesNode->FirstChildElement( "object" ); itemNode; itemNode = itemNode->NextSiblingElement( "object" ) )
        ++objectCount_;

    roots_.push_back( std::make_shared<Node>( xmlNode ) );
    if ( const auto res = roots_.back()->load(); !res )
        return unexpected( res.error() );

    ++documentsLoaded_;
    return {};
}

Expected<void> ThreeMFLoader::loadTree_( ProgressCallback callback )
{
    roots_.reserve( documents_.size() );
    Node::loader = this;

    for ( size_t i = 0; i < documents_.size(); ++i )
    {
        if ( auto resOrErr = loadDocument_( documents_[i], subprogress(callback, documentsLoaded_, documents_.size())); !resOrErr )
            return unexpected( resOrErr.error() );
    }

    return {};
}

Expected<std::shared_ptr<Object>> ThreeMFLoader::load( const std::vector<std::filesystem::path>& files, std::filesystem::path root, ProgressCallback callback )
{
    rootPath_ = root.lexically_normal();

    auto res = loadXmls_( files );
    if ( !res )
        return unexpected( res.error() );

    if ( !reportProgress( callback, 0.2f ) )
        return unexpected( std::string( "Loading canceled" ) );

    res = loadTree_( subprogress( callback, 0.2f, 0.8f ) );

    if ( !res )
        return unexpected( res.error() );

    if ( !reportProgress( callback, 0.8f ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( objectNodes_.empty() )
        return unexpected( "No objects found" );

    size_t unnamedMeshCounter = 0;

    std::shared_ptr<Object> objRes = std::make_shared<Object>();
    for ( auto& node : objectNodes_ )
    {
        std::shared_ptr<ObjectMesh> objMesh = std::make_shared<ObjectMesh>();
        objMesh->setMesh( std::make_shared<Mesh>( std::move( node->mesh ) ) );
        objMesh->setXf( node->xf );
        if ( !node->objName.empty() )
            objMesh->setName( node->objName );
        else
            objMesh->setName( fmt::format( "Mesh {}", ++unnamedMeshCounter ) );
        objMesh->setFrontColor( node->bgColor, false );

        if ( node->texId != -1 )
        {            
            //if any vertex has NaN UV, we can't load the texture
            if ( std::find_if( node->vertUVCoords.vec_.begin(), node->vertUVCoords.vec_.end(), [] ( const auto& uv ) { return std::isnan( uv.x ); } ) == node->vertUVCoords.vec_.end() )
            { 
                auto it = idToNodeMap_.find( node->texId );
                if ( it == idToNodeMap_.end() )
                    return unexpected( "Invalid texture id" );

                if ( it->second->texture.resolution.x > 0 && it->second->texture.resolution.y > 0 )
                {
                    //cannot move because the same texture could be used in multiple objects
                    objMesh->setTextures( { it->second->texture } );
                    objMesh->setUVCoords( std::move( node->vertUVCoords ) );
                    objMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                }
                else if ( loadWarn )
                {
                    loadWarn->append( "Texture will not be loaded.\n" );
                }
            }
            else if ( loadWarn )
            {
                loadWarn->append(  "Object" + node->objName + " has incomplete UV coordinates. Texture will not be loaded.\n" );
            }
        }

        if ( !node->vertColorMap.empty() )
        {
            objMesh->setVertsColorMap( std::move( node->vertColorMap ) );
            objMesh->setColoringType( ColoringType::VertsColorMap );
        }

        objRes->addChild( std::move( objMesh ) );
    }

    if ( !reportProgress( callback, 1.0f ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( duplicatedVertexCountAccum > 0 && loadWarn )
        loadWarn->append( "Duplicated vertex count: " + std::to_string( duplicatedVertexCountAccum ) + "\n" );

    if ( skippedFaceCountAccum > 0 && loadWarn )
        loadWarn->append( "Skipped face count: " + std::to_string( skippedFaceCountAccum ) + "\n" );

    return objRes->children().size() > 1 ? objRes : objRes->children().front();
}

Expected<void> Node::loadObject_( const tinyxml2::XMLElement* xmlNode, ProgressCallback callback )
{
    if ( auto refNode = pNode; refNode && ( refNode->nodeType == NodeType::ColorGroup || refNode->nodeType == NodeType::BaseMaterials ) )
    {
        if ( pindex < 0 || pindex >= refNode->colors.size() )
            return unexpected( "Invalid color index" );

        bgColor = refNode->colors[pindex];
    }

    auto meshNode = xmlNode->FirstChildElement( "mesh" );
    auto componentsNode = xmlNode->FirstChildElement( "components" );
    if ( meshNode )
    {
        auto meshErr = loadMesh_( meshNode, callback );
        if ( !meshErr )
            return unexpected( meshErr.error() );

        auto nameAttr = xmlNode->Attribute( "name" );
        if ( nameAttr )
            objName = std::string( nameAttr );

        mesh = std::move( meshErr.value() );
        return {};
    }
    else if ( componentsNode )
    {
        for ( auto componentNode = componentsNode->FirstChildElement( "component" ); componentNode; componentNode = componentNode->NextSiblingElement( "component" ) )
        {
            int objId = -1;
            if ( tinyxml2::XML_SUCCESS != componentNode->QueryIntAttribute( "objectid", &objId ) )
                return unexpected( "Invalid object id" );

            AffineXf3f transform;
            auto transformAttr = componentNode->Attribute( "transform" );
            if ( transformAttr )
            {
                auto xfRes = parseAffineXf( transformAttr );
                if ( !xfRes )
                    return unexpected( xfRes.error() );
                transform = std::move( *xfRes );
            }

            auto it = loader->idToNodeMap_.find( objId );
            if ( it == loader->idToNodeMap_.end() )
            {
                auto pathAttr = componentNode->Attribute( "p:path" );
                if ( !pathAttr )
                    return unexpected( "Invalid 'p:path attribute'" );

                std::filesystem::path path = loader->rootPath_ /  ("./" + std::string( pathAttr ) );
                auto docRes = loader->loadXml_(  path );
                if ( !docRes )
                    return unexpected( docRes.error() );

                auto doc = std::move( *docRes );
                loader->loadDocument_( doc, subprogress( loader->generalCallback, loader->documentsLoaded_, loader->documents_.size() ) );

                it = loader->idToNodeMap_.find( objId );
                if ( it == loader->idToNodeMap_.end() )
                    return unexpected( "Invalid object id" );
            }

            bgColor = it->second->bgColor;
            if ( transform == AffineXf3f() )
            {
                mesh.addPart( it->second->mesh );
            }
            else
            {
                auto meshCopy = it->second->mesh;
                meshCopy.transform( transform );
                mesh.addPart( std::move( meshCopy ) );
            }
        }
        return {};
    }
    return unexpected( "Object has no mesh" );
}

Expected<void> Node::loadBuildData_( const tinyxml2::XMLElement* xmlNode )
{
    for ( auto itemNode = xmlNode->FirstChildElement( "item" ); itemNode; itemNode = itemNode->NextSiblingElement( "item" ) )
    {
        auto objIdAttr = itemNode->Attribute( "objectid" );
        if ( !objIdAttr )
            continue;

        const int objId = std::stoi( objIdAttr );
        auto it = loader->idToNodeMap_.find( objId );
        if ( it == loader->idToNodeMap_.end() )
        {
            auto pathAttr = itemNode->Attribute( "p:path" );
            if ( !pathAttr )
                return unexpected( "Invalid 'p:path' attribute" );

            std::filesystem::path path = loader->rootPath_ / ( "./" + std::string( pathAttr ) );
            auto docRes = loader->loadXml_( path );
            if ( !docRes )
                return unexpected( docRes.error() );

            auto doc = std::move( *docRes );
            loader->loadDocument_( doc, subprogress( loader->generalCallback, loader->documentsLoaded_, loader->documents_.size() ) );

            it = loader->idToNodeMap_.find( objId );
            if ( it == loader->idToNodeMap_.end() )
                return unexpected( "Invalid object id" );
        }

        auto objNode = it->second;
        auto transformAttr = itemNode->Attribute( "transform" );
        if ( transformAttr )
        {
            auto resXf = parseAffineXf( std::string( transformAttr ) );
            if ( !resXf )
                return unexpected( resXf.error() );

            if ( resXf->A.det() == 0 && loader->loadWarn )
                loader->loadWarn->append( "Degenerative object transform: " + objNode->objName + "\n" );

            objNode->xf = *resXf;
        }

        loader->objectNodes_.push_back( objNode );
    }

    return {};
}

Expected<void> Node::load()
{
    if ( auto it = nodeTypeMap.find( nodeName ); it != nodeTypeMap.end() )
        nodeType = it->second;

    auto attr = node->Attribute( "id" );
    if ( attr )
    {
        id = std::stoi( attr );
        loader->idToNodeMap_[id] = this;
    }

    attr = node->Attribute( "pid" );
    if ( attr )
    {
        pid = std::stoi( attr );        
        if ( auto it = loader->idToNodeMap_.find( pid ); it != loader->idToNodeMap_.end() )
            pNode = it->second;
    }

    attr = node->Attribute( "pindex" );
    if ( attr )
        pindex = std::stoi( attr );

    switch ( nodeType )
    {
    case NodeType::ColorGroup:
        if ( auto res = loadColorGroup_( node ); !res )
            return unexpected( res.error() );
        break;    
    case NodeType::BaseMaterials:
        if ( auto res = loadBaseMaterials_( node ); !res )
            return unexpected( res.error() );
        break;
    case NodeType::Object:
        if ( auto res = loadObject_( node, subprogress( loader->documentProgressCallback, loader->objectsLoaded_, loader->objectCount_ ) ); !res )
            return unexpected( res.error() );
        break;
    case NodeType::Build:
        if ( auto res = loadBuildData_( node ); !res )
            return unexpected( res.error() );
        break;
    case NodeType::Texture2d:
        if ( auto res = loadTexture2d_( node ); !res && loader->loadWarn  && loader->loadWarn->empty() )
            loader->loadWarn->append( res.error() );
        break;
    case NodeType::Texture2dGroup:
        if ( auto res = loadTexture2dGroup_( node ); !res && loader->loadWarn )
            loader->loadWarn->append( res.error() );
        break;
    case NodeType::Multiproperties:
        if ( auto res = loadMultiproperties_( node ); !res )
            return unexpected( res.error() );
        break;
    default:    
        for ( auto childNode = node->FirstChildElement(); childNode; childNode = childNode->NextSiblingElement() )
        {
            children.push_back( std::make_shared<Node>( childNode ) );
            if ( auto resOrErr = children.back()->load(); !resOrErr )
                return unexpected( resOrErr.error() );
        }
        break;    
    }

    return {};
}

Expected<void> Node::loadBaseMaterials_( const tinyxml2::XMLElement* xmlNode )
{
    for ( auto materialNode = xmlNode->FirstChildElement( "base" ); materialNode; materialNode = materialNode->NextSiblingElement() )
    {
        const auto colorStr = std::string( materialNode->Attribute( "displaycolor" ) );
        auto colorOrErr = parseColor( colorStr );
        if ( !colorOrErr )
            return unexpected( colorOrErr.error() );

        colors.push_back( std::move( colorOrErr.value() ) );
    }

    return {};
}

Expected<void> Node::loadColorGroup_( const tinyxml2::XMLElement* xmlNode )
{
    for ( auto colorNode = xmlNode->FirstChildElement( "m:color" ); colorNode; colorNode = colorNode->NextSiblingElement( "m:color" ) )
    {
        const auto colorStr = std::string( colorNode->Attribute( "color" ) );
        auto colorOrErr = parseColor( colorStr );
        if ( !colorOrErr )
            return unexpected( colorOrErr.error() );

        colors.push_back( std::move( colorOrErr.value() ) );
    }

    return {};
}

Expected<void> Node::loadTexture2d_( const tinyxml2::XMLElement* xmlNode )
{
    std::string innerPath = "./" + std::string(xmlNode->Attribute("path"));
    if ( innerPath.size() == 2 )
        return unexpected( std::string( "Texture2d node does not have 'path' attribute" ) );

    std::filesystem::path fullPath = loader->rootPath_ / innerPath;
    std::error_code ec;
    if ( !std::filesystem::exists( fullPath, ec ) )
        return unexpected( std::string( "Texture2d does not exist: " ) + utf8string( fullPath ) );

    auto imageExp = ImageLoad::fromAnySupportedFormat( fullPath );
    if ( !imageExp )
        return unexpected( imageExp.error() );

    *static_cast<Image*>( &texture ) = std::move( *imageExp );
    texture.wrap = WrapType::Repeat;
    texture.filter = FilterType::Linear;

    return {};
}

Expected<void> Node::loadTexture2dGroup_( const tinyxml2::XMLElement* xmlNode )
{
    if ( tinyxml2::XML_SUCCESS != xmlNode->QueryIntAttribute( "texid", &texId ) )
        return unexpected( std::string( "3DF model texture2d group node does not have 'texid' attribute" ) );

    auto it = loader->idToNodeMap_.find( texId );
    if ( it == loader->idToNodeMap_.end() )
        return unexpected( std::string( "3DF model has incorrect 'texid' attribute" ) );

    for ( auto coordNode = xmlNode->FirstChildElement( "m:tex2coord" ); coordNode; coordNode = coordNode->NextSiblingElement( "m:tex2coord" ) )
    {
        uvCoords.emplace_back();
        auto& uv = uvCoords.back();
        if ( tinyxml2::XML_SUCCESS != coordNode->QueryFloatAttribute( "u", &uv.x ) )
            return unexpected( std::string( "3DF model tex2coord node does not have 'u' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != coordNode->QueryFloatAttribute( "v", &uv.y ) )
            return unexpected( std::string( "3DF model tex2coord node does not have 'v' attribute" ) );
    }
    return {};
}

Expected<Mesh> Node::loadMesh_( const tinyxml2::XMLElement* meshNode, ProgressCallback callback )
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

        int texGroupId = -1;
        triangleNode->QueryIntAttribute( "pid", &texGroupId );
        if ( texGroupId == -1 )
            continue;
        
        auto it = loader->idToNodeMap_.find( texGroupId );
        if ( it == loader->idToNodeMap_.end() || 
        ( it->second->nodeType != NodeType::Texture2dGroup 
        && it->second->nodeType != NodeType::ColorGroup 
        && it->second->nodeType != NodeType::BaseMaterials
        && it->second->nodeType != NodeType::Multiproperties ) )
        {
            if ( !loader->failedToLoadColoring )
            {
                if ( loader->loadWarn )
                    loader->loadWarn->append( std::string( "3DF model has unsupported coloring\n" ) );

                loader->failedToLoadColoring = true;               
            }
            continue;
        }

        if ( it->second->nodeType == NodeType::Texture2dGroup )
        {
            texId = it->second->texId;
            if ( vertUVCoords.empty() )
                // fill vector with NaN in order to identify vertices without UV coordinates. If any vertex has no UV coordinates, texture will be ignored
                vertUVCoords.resize( vertexCoordinates.size(), { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN() } );
        }
        else if ( ( it->second->nodeType == NodeType::ColorGroup || it->second->nodeType == NodeType::BaseMaterials ) && vertColorMap.empty() )
        {
            vertColorMap.resize( vertexCoordinates.size(), bgColor );
        }
        else if ( it->second->nodeType == NodeType::Multiproperties )
        {
            auto pidIt = std::find_if( it->second->pids.begin(), it->second->pids.end(), [&] ( int pid )
            {
                auto it = loader->idToNodeMap_.find( pid );
                if ( it == loader->idToNodeMap_.end() )
                    return false;

                return it->second->nodeType == NodeType::Texture2dGroup;
            } );

            if ( pidIt != it->second->pids.end() )
                texId = loader->idToNodeMap_[*pidIt]->texId;

            if ( vertColorMap.empty() )
                vertColorMap.resize( vertexCoordinates.size(), bgColor );

            if ( vertUVCoords.empty() )
                vertUVCoords.resize( vertexCoordinates.size(), { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN() } );
        }

        int ps[3] = {};
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "p1", &ps[0] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'p1' attribute" ) );

        if ( triangleNode->Attribute( "p2" ) )
        {
            if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "p2", &ps[1] ) )
                return unexpected( std::string( "3DF model triangle node does not have 'p2' attribute" ) );
            if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "p3", &ps[2] ) )
                return unexpected( std::string( "3DF model triangle node does not have 'p3' attribute" ) );
        }
        else
        {
            ps[1] = ps[0];
            ps[2] = ps[0];
        }

        if ( it->second->nodeType == NodeType::Texture2dGroup )
        {
            for ( int i = 0; i < 3; ++i )
            {
                if ( vs[i] < 0 || vs[i] >= vertUVCoords.size() )
                    return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );               

                if ( ps[i] < 0 || ps[i] >= it->second->uvCoords.size() )
                    return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                auto& vertUV = vertUVCoords[VertId( vs[i] )];
                const auto refUV = it->second->uvCoords[ps[i]];
                // If vertex already has another UV coordinates, texture will be ignored
                if ( !loader->failedToLoadColoring && !std::isnan( vertUV.x ) && ( vertUV.x != refUV.x || vertUV.y != refUV.y ) )
                {
                    loader->failedToLoadColoring = true;
                    if ( loader->loadWarn )
                        loader->loadWarn->append( "Texture cannot be show correctly because 3DF model has different UV coordinates for some vertices\n" );
                }

                vertUV = refUV;
            }
        }
        else if ( it->second->nodeType == NodeType::ColorGroup )
        {
            for ( int i = 0; i < 3; ++i )
            {
                if ( ps[i] < 0 || ps[i] >= it->second->colors.size() )
                    return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                vertColorMap[VertId( vs[i] )] = it->second->colors[ps[i]];
            }
        }
        else //it->second->nodeType == NodeType::Multiproperties
        {
            const auto& refPids = it->second->pids;
            Node* colorNode = nullptr;
            int colorIndex = -1;
            Node* textureNode = nullptr;
            int textureIndex = -1;
            
            auto colorIt = std::find_if( refPids.begin(), refPids.end(), [&] ( int pid )
            {
                auto it = loader->idToNodeMap_.find( pid );
                return ( it == loader->idToNodeMap_.end() ) ? false : it->second->nodeType == NodeType::ColorGroup;
            } );

            if ( colorIt != refPids.end() )
            { 
                colorNode = loader->idToNodeMap_[*colorIt];
                colorIndex = int( colorIt - refPids.begin() );
            }

            auto textureIt = std::find_if( refPids.begin(), refPids.end(), [&] ( int pid )
            {
                auto it = loader->idToNodeMap_.find( pid );
                return ( it == loader->idToNodeMap_.end() ) ? false : it->second->nodeType == NodeType::Texture2dGroup;
            } );

            if ( textureIt != refPids.end() )
            { 
                textureNode = loader->idToNodeMap_[*textureIt];
                textureIndex = int( textureIt - refPids.begin() );
            }

            if ( colorNode && !loader->failedToLoadColoring )
            {
                for ( int i = 0; i < 3; ++i )
                {
                    if ( ps[i] < 0 || ps[i] >= it->second->pindices.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    if ( vs[i] < 0 || vs[i] >= vertColorMap.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );

                    const auto& refPindices = it->second->pindices[ps[i]];

                    if ( refPindices[colorIndex] < 0 || refPindices[colorIndex] >= colorNode->colors.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );
                    vertColorMap[VertId( vs[i] )] = colorNode->colors[refPindices[colorIndex]];
                }
            }

            if ( textureNode && !loader->failedToLoadColoring )
            {

                for ( int i = 0; i < 3; ++i )
                {
                    if ( ps[i] < 0 || ps[i] >= it->second->pindices.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    if ( vs[i] < 0 || vs[i] >= vertUVCoords.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );
                    const auto& refPindices = it->second->pindices[ps[i]];
                    if ( refPindices[textureIndex] < 0 || refPindices[textureIndex] >= textureNode->uvCoords.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    auto& vertUV = vertUVCoords[VertId( vs[i] )];
                    const auto refUV = textureNode->uvCoords[refPindices[textureIndex]];
                    // If vertex already has another UV coordinates, texture will be ignored
                    if ( !std::isnan( vertUV.x ) && ( vertUV.x != refUV.x || vertUV.y != refUV.y ) )
                    {
                        loader->failedToLoadColoring = true;
                        if ( loader->loadWarn )
                            loader->loadWarn->append( "Texture cannot be show correctly because 3DF model has different UV coordinates for some vertices\n" );
                    }

                    vertUV = refUV;
                }
            }
        }        
    }

    if ( !reportProgress( callback, 0.5f ) )
        return unexpected( std::string( "Loading canceled" ) );

    std::vector<MeshBuilder::VertDuplication> dups;
    int skippedFaceCount = 0;
    MeshBuilder::BuildSettings buildSettings{ .skippedFaceCount = &skippedFaceCount };

    MR::Mesh res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris, &dups, buildSettings );
    
    if ( !dups.empty() )
    {
        loader->duplicatedVertexCountAccum += int( dups.size() );
        if ( !vertUVCoords.empty() )
        {
            vertUVCoords.resize( res.topology.lastValidVert() + 1 );
            for ( auto& dup : dups )
            {
                vertUVCoords[dup.dupVert] = vertUVCoords[dup.srcVert];
            }
        }

        if ( !vertColorMap.empty() )
        {
            vertColorMap.resize( res.topology.lastValidVert() + 1 );
            for ( auto& dup : dups )
            {
                vertColorMap[dup.dupVert] = vertColorMap[dup.srcVert];
            }
        }
    }
    
    loader->skippedFaceCountAccum += skippedFaceCount;

    if ( !reportProgress( callback, 0.75f ) )
        return unexpected( std::string( "Loading canceled" ) );

    return res;
}

Expected<void> Node::loadMultiproperties_( const tinyxml2::XMLElement* xmlNode )
{
    std::string attr( xmlNode->Attribute( "pids" ) );
    auto intsOrErr = parseInts( attr );
    if ( !intsOrErr )
        return unexpected( intsOrErr.error() );

    pids = std::move( *intsOrErr );
    const size_t pidCount = pids.size();

    for ( auto multiNode = xmlNode->FirstChildElement( "m:multi" ); multiNode; multiNode = multiNode->NextSiblingElement( "m:multi" ) )
    {
        attr = std::string( multiNode->Attribute( "pindices" ) );
        intsOrErr = parseInts( attr );
        if ( !intsOrErr )
            return unexpected( intsOrErr.error() );
        
        pindices.emplace_back();

        pindices.back() = std::move(*intsOrErr);
        if ( pindices.back().size() != pidCount )
        {
            pindices.back().resize( pidCount, pindices.back().front() );
        }
        
    }

    return {};
}

Expected<std::shared_ptr<Object>> deserializeObjectTreeFrom3mf( const std::filesystem::path& path, std::string* loadWarn, ProgressCallback callback )
{
    const auto tmpFolder = UniqueTemporaryFolder( {} );

    auto resZip = decompressZip( path, tmpFolder );
    if ( !resZip )
        return unexpected( "ZIP container error: " + resZip.error() );

    if ( !reportProgress( callback, 0.1f ) )
        return unexpected( std::string( "Loading canceled" ) );

    std::vector<std::filesystem::path> files;
    std::error_code ec;

    for ( auto const& dirEntry : DirectoryRecursive{ tmpFolder, ec } )
        if ( !dirEntry.is_directory( ec ) )
            files.push_back( dirEntry.path() );

    if ( files.empty() )
        return unexpected( "Could not find .model" );

    ThreeMFLoader loader;
    loader.loadWarn = loadWarn;
    return loader.load( files, tmpFolder, subprogress( callback, 0.1f, 0.9f ) );
}

Expected<std::shared_ptr<Object>> deserializeObjectTreeFromModel( const std::filesystem::path& path, std::string* loadWarn, ProgressCallback callback )
{
    ThreeMFLoader loader;
    loader.loadWarn = loadWarn;
    return loader.load( { path }, path.parent_path(), callback );
}

MR_ADD_SCENE_LOADER( IOFilter( "3D Manufacturing format (.3mf)", "*.3mf" ), deserializeObjectTreeFrom3mf )
MR_ADD_SCENE_LOADER( IOFilter( "3D Manufacturing model (.model)", "*.model" ), deserializeObjectTreeFromModel )

}
#endif
