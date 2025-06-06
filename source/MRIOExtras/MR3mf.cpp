#include "MR3mf.h"
#ifndef MRIOEXTRAS_NO_3MF

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
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
#include "MRPch/MRJson.h"

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
    
    Expected<void> loadMesh_( const tinyxml2::XMLElement* meshNode, ProgressCallback callback );

    int id = -1;
    int pid = -1;
    int pindex = -1;
    int texId = -1;

    Node* pNode = nullptr;

    NodeType nodeType = NodeType::Unknown;
    std::vector<std::shared_ptr<Node>> children;
    std::string nodeName;

    MeshTexture texture;
    std::vector<UVCoord> uvGroup;
    std::shared_ptr<Object> obj;
    std::vector<Color> colors;

    void setFilamentFaceColor_( FaceId f, const std::string& id, FaceColors& fColorMap );


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
    struct LoadedXml
    {
        std::unique_ptr<tinyxml2::XMLDocument> doc;
        bool loaded{ false };
    };
    std::unordered_map<std::string, LoadedXml> xmlDocuments_;
    std::unordered_map<std::string, Json::Value> jsonDocuments_;
    std::filesystem::path rootPath_;    
    // Object tree - each node is either a mesh or compound object

    std::vector<Node*> objectNodes_;

    std::vector<std::shared_ptr<Node>> roots_;

    Expected<std::unique_ptr<tinyxml2::XMLDocument>> loadXml_( const std::filesystem::path& file );
    // Load and parse all XML .model files
    Expected<void> loadFiles_( const std::vector<std::filesystem::path>& files );

    // Load object tree from loaded XML files
    Expected<void> loadTree_( ProgressCallback callback );
    Expected<void> loadDocument_( LoadedXml& doc, ProgressCallback callback );

    int duplicatedVertexCountAccum = 0;
    int skippedFaceCountAccum = 0;    
    
    ProgressCallback documentProgressCallback;
    ProgressCallback generalCallback;

    size_t objectCount_ = 0;
    size_t objectsLoaded_ = 0;
    size_t documentsLoaded_ = 0;

    std::unordered_map<int, Node*> nodeByIdMap_;
    Expected<Node*> getNodeById_( int id, const char* path = nullptr );

    std::vector<Color> filamentColors;
    bool filamentInited_{ false };
    void initFilamentColors_(); // specific way for FaceColors used in PrunaSlicer and BambuStudio
public:
    std::string warnings;
    bool failedToLoadColoring = false;

    Expected<LoadedObject> load( const std::vector<std::filesystem::path>& files, std::filesystem::path root, ProgressCallback callback );

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

Expected<void> ThreeMFLoader::loadFiles_( const std::vector<std::filesystem::path>& files )
{
    for ( std::filesystem::path file : files )
    {
        auto docRes = loadXml_( file );
        if ( !docRes )
            return unexpected( docRes.error() );

        if ( *docRes != nullptr )
        {
            // Store parsed XML
            xmlDocuments_[utf8string( file.lexically_normal() )] = { .doc = std::move( *docRes ) };
        }
        else
        {
            // Try load as Json
            auto jsonRes = deserializeJsonValue( file );
            if ( !jsonRes.has_value() )
                continue;
            jsonDocuments_[utf8string( file.lexically_normal() )] = std::move( *jsonRes );
        }
    }
    return {};
}

Expected<void> ThreeMFLoader::loadDocument_( LoadedXml& doc, ProgressCallback callback )
{
    if ( doc.loaded )
        return {};
    doc.loaded = true;
    auto xmlNode = doc.doc->FirstChildElement();
    if ( std::string( xmlNode->Name() ) != "model" ) //maybe another xml, just skip
        return {};

    objectCount_ = 0;
    objectsLoaded_ = 0;
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

Expected<Node*> ThreeMFLoader::getNodeById_( int id, const char* pathAttr )
{
    auto it = nodeByIdMap_.find( id );
    if ( it != nodeByIdMap_.end() )
        return it->second;

    if ( !pathAttr )
        return unexpected( "Invalid 'p:path attribute'" );

    std::filesystem::path path = rootPath_ / ( "./" + std::string( pathAttr ) );
    auto docIt = xmlDocuments_.find( utf8string( path.lexically_normal() ) );
    if ( docIt == xmlDocuments_.end() )
        return unexpected( "Cannot find file specified in p:path" );

    if ( auto e = loadDocument_( docIt->second, subprogress( generalCallback, documentsLoaded_, xmlDocuments_.size() ) ); !e )
        return unexpected( std::move( e.error() ) );

    it = nodeByIdMap_.find( id );
    if ( it != nodeByIdMap_.end() )
        return it->second;
    return unexpected( "Invalid object id" );
}

Expected<void> ThreeMFLoader::loadTree_( ProgressCallback callback )
{
    roots_.reserve( xmlDocuments_.size() );
    Node::loader = this;

    initFilamentColors_();

    for ( auto& [_, xmlDoc] : xmlDocuments_ )
    {
        if ( auto resOrErr = loadDocument_( xmlDoc, subprogress(callback, documentsLoaded_, xmlDocuments_.size())); !resOrErr )
            return unexpected( resOrErr.error() );
    }

    return {};
}

Expected<LoadedObject> ThreeMFLoader::load( const std::vector<std::filesystem::path>& files, std::filesystem::path root, ProgressCallback callback )
{
    rootPath_ = root.lexically_normal();

    auto maybe = loadFiles_( files );
    if ( !maybe )
        return unexpected( std::move( maybe.error() ) );

    if ( !reportProgress( callback, 0.2f ) )
        return unexpectedOperationCanceled();

    maybe = loadTree_( subprogress( callback, 0.2f, 0.8f ) );

    if ( !maybe )
        return unexpected( std::move( maybe.error() ) );

    if ( !reportProgress( callback, 0.8f ) )
        return unexpectedOperationCanceled();

    if ( objectNodes_.empty() )
        return unexpected( "No objects found" );

    std::shared_ptr<Object> objRes = std::make_shared<Object>();
    for ( auto& node : objectNodes_ )
    {
        objRes->addChild( std::move( node->obj ) );
    }

    if ( !reportProgress( callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( duplicatedVertexCountAccum > 0 )
        warnings.append( "Duplicated vertex count: " + std::to_string( duplicatedVertexCountAccum ) + "\n" );

    if ( skippedFaceCountAccum > 0 )
        warnings.append( "Skipped face count: " + std::to_string( skippedFaceCountAccum ) + "\n" );

    if ( objRes->children().size() == 1 )
        objRes = objRes->children()[0];
    return LoadedObject{ .obj = objRes, .warnings = std::move( warnings ) };
}

Expected<void> Node::loadObject_( const tinyxml2::XMLElement* xmlNode, ProgressCallback callback )
{
    auto meshNode = xmlNode->FirstChildElement( "mesh" );
    auto componentsNode = xmlNode->FirstChildElement( "components" );
    if ( meshNode )
    {
        auto meshErr = loadMesh_( meshNode, callback );
        if ( !meshErr )
            return unexpected( meshErr.error() );
    }
    else if ( componentsNode )
    {
        obj = std::make_shared<Object>();
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

            auto nodeRes = loader->getNodeById_( objId, componentNode->Attribute( "p:path" ) );
            if ( !nodeRes.has_value() )
                return unexpected( std::move( nodeRes.error() ) );

            auto* nodePtr = *nodeRes;
            assert( nodePtr );

            if ( !nodePtr->obj )
            {
                assert( false );
                return unexpected( "Incorrect loading sequence" );
            }
            if ( !nodePtr->obj->parent() )
            {
                nodePtr->obj->setXf( transform );
                obj->addChild( nodePtr->obj );
            }
            else
            {
                auto cloneTree = nodePtr->obj->cloneTree();
                cloneTree->setXf( transform );
                obj->addChild( cloneTree );
            }
        }
        return {};
    }
    else
    {
        return unexpected( "Object has no mesh" );
    }
    auto nameAttr = xmlNode->Attribute( "name" );
    if ( nameAttr )
    {
        assert( obj );
        obj->setName( nameAttr );
    }
    else
    {
        obj->setName( "Object_" + std::to_string( id ) );
    }
    obj->select( true );
    return {};
}

Expected<void> Node::loadBuildData_( const tinyxml2::XMLElement* xmlNode )
{
    for ( auto itemNode = xmlNode->FirstChildElement( "item" ); itemNode; itemNode = itemNode->NextSiblingElement( "item" ) )
    {
        auto objIdAttr = itemNode->Attribute( "objectid" );
        if ( !objIdAttr )
            continue;

        const int objId = std::stoi( objIdAttr );

        auto nodeRes = loader->getNodeById_( objId, itemNode->Attribute( "p:path" ) );
        if ( !nodeRes.has_value() )
            return unexpected( std::move( nodeRes.error() ) );

        auto* objNode = *nodeRes;
        assert( objNode );
        assert( objNode->obj );

        auto transformAttr = itemNode->Attribute( "transform" );
        if ( transformAttr )
        {
            auto resXf = parseAffineXf( std::string( transformAttr ) );
            if ( !resXf )
                return unexpected( resXf.error() );

            if ( resXf->A.det() == 0 )
                loader->warnings.append( "Degenerate object transform: " + objNode->obj->name() + "\n" );

            objNode->obj->setXf( *resXf );
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
        loader->nodeByIdMap_[id] = this;
    }

    attr = node->Attribute( "pid" );
    if ( attr )
    {
        pid = std::stoi( attr );        
        if ( auto it = loader->nodeByIdMap_.find( pid ); it != loader->nodeByIdMap_.end() )
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
        if ( auto res = loadObject_( node, subprogress( loader->documentProgressCallback, loader->objectsLoaded_++, loader->objectCount_ ) ); !res )
            return unexpected( res.error() );
        break;
    case NodeType::Build:
        if ( auto res = loadBuildData_( node ); !res )
            return unexpected( res.error() );
        break;
    case NodeType::Texture2d:
        if ( auto res = loadTexture2d_( node ) )
            loader->warnings.append( res.error() + '\n' );
        break;
    case NodeType::Texture2dGroup:
        if ( auto res = loadTexture2dGroup_( node ) )
            loader->warnings.append( res.error() + '\n' );
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

    auto nodeRes = loader->getNodeById_( texId );
    if ( !nodeRes.has_value() )
        return unexpected( std::string( "3DF model has incorrect 'texid' attribute" ) );

#if TINYXML2_MAJOR_VERSION > 10
    uvGroup.reserve( xmlNode->ChildElementCount( "m:tex2coord" ) );
#endif
    for ( auto coordNode = xmlNode->FirstChildElement( "m:tex2coord" ); coordNode; coordNode = coordNode->NextSiblingElement( "m:tex2coord" ) )
    {
        auto& uv = uvGroup.emplace_back();
        if ( tinyxml2::XML_SUCCESS != coordNode->QueryFloatAttribute( "u", &uv.x ) )
            return unexpected( std::string( "3DF model tex2coord node does not have 'u' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != coordNode->QueryFloatAttribute( "v", &uv.y ) )
            return unexpected( std::string( "3DF model tex2coord node does not have 'v' attribute" ) );
    }
    return {};
}

Expected<void> Node::loadMesh_( const tinyxml2::XMLElement* meshNode, ProgressCallback callback )
{
    auto verticesNode = meshNode->FirstChildElement( "vertices" );
    if ( !verticesNode )
        return unexpected( std::string( "3DF model 'vertices' node not found" ) );    

    Color bgColor = Color::white();
    bool bgColorWasRead = false;
    if ( auto refNode = pNode; refNode && ( refNode->nodeType == NodeType::ColorGroup || refNode->nodeType == NodeType::BaseMaterials ) )
    {
        if ( pindex < 0 || pindex >= refNode->colors.size() )
            return unexpected( "Invalid color index" );

        bgColorWasRead = true;
        bgColor = refNode->colors[pindex];
    }

    VertCoords vertexCoordinates;
#if TINYXML2_MAJOR_VERSION > 10
    vertexCoordinates.reserve( verticesNode->ChildElementCount( "vertex" ) );
#endif
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
        return unexpectedOperationCanceled();

    auto trianglesNode = meshNode->FirstChildElement( "triangles" );
    if ( !trianglesNode )
        return unexpected( std::string( "3DF model 'triangles' node not found" ) );

    FaceColors fColorMap;
    VertColors vColorMap;
    VertUVCoords vUVCoords;

    Triangulation tris;
#if TINYXML2_MAJOR_VERSION > 10
    tris.reserve( trianglesNode->ChildElementCount( "triangle" ) );
#endif
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
        {
            // try load filament color used in PrunaSlicer or BambuStudio
            const char* filamentId = nullptr;
            auto hasFilamentId = triangleNode->QueryStringAttribute( "slic3rpe:mmu_segmentation", &filamentId );
            if ( hasFilamentId == tinyxml2::XML_NO_ATTRIBUTE )
                hasFilamentId = triangleNode->QueryStringAttribute( "paint_color", &filamentId );
            if ( hasFilamentId == tinyxml2::XML_SUCCESS )
                setFilamentFaceColor_( tris.backId(), std::string( filamentId ), fColorMap );
            continue;
        }
        
        auto it = loader->nodeByIdMap_.find( texGroupId );
        if ( it == loader->nodeByIdMap_.end() ||
        ( it->second->nodeType != NodeType::Texture2dGroup 
        && it->second->nodeType != NodeType::ColorGroup 
        && it->second->nodeType != NodeType::BaseMaterials
        && it->second->nodeType != NodeType::Multiproperties ) )
        {
            if ( !loader->failedToLoadColoring )
            {
                loader->warnings.append( std::string( "3DF model has unsupported coloring\n" ) );
                loader->failedToLoadColoring = true;
            }
            continue;
        }

        if ( it->second->nodeType == NodeType::Texture2dGroup )
        {
            texId = it->second->texId;
            if ( vUVCoords.empty() )
                // fill vector with NaN in order to identify vertices without UV coordinates. If any vertex has no UV coordinates, texture will be ignored
                vUVCoords.resize( vertexCoordinates.size(), { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN() } );
        }
        else if ( ( it->second->nodeType == NodeType::ColorGroup || it->second->nodeType == NodeType::BaseMaterials ) && vColorMap.empty() )
        {
            vColorMap.resize( vertexCoordinates.size(), bgColor );
        }
        else if ( it->second->nodeType == NodeType::Multiproperties )
        {
            auto pidIt = std::find_if( it->second->pids.begin(), it->second->pids.end(), [&] ( int pid )
            {
                auto it = loader->nodeByIdMap_.find( pid );
                if ( it == loader->nodeByIdMap_.end() )
                    return false;

                return it->second->nodeType == NodeType::Texture2dGroup;
            } );

            if ( pidIt != it->second->pids.end() )
                texId = loader->nodeByIdMap_[*pidIt]->texId;

            if ( vColorMap.empty() )
                vColorMap.resize( vertexCoordinates.size(), bgColor );

            if ( vUVCoords.empty() )
                vUVCoords.resize( vertexCoordinates.size(), { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN() } );
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
                if ( vs[i] < 0 || vs[i] >= vUVCoords.size() )
                    return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );               

                if ( ps[i] < 0 || ps[i] >= it->second->uvGroup.size() )
                    return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                auto& vertUV = vUVCoords[VertId( vs[i] )];
                const auto refUV = it->second->uvGroup[ps[i]];
                // If vertex already has another UV coordinates, texture will be ignored
                if ( !loader->failedToLoadColoring && !std::isnan( vertUV.x ) && ( vertUV.x != refUV.x || vertUV.y != refUV.y ) )
                {
                    loader->failedToLoadColoring = true;
                    loader->warnings.append( "Texture cannot be show correctly because 3DF model has different UV coordinates for some vertices\n" );
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

                vColorMap[VertId( vs[i] )] = it->second->colors[ps[i]];
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
                auto it = loader->nodeByIdMap_.find( pid );
                return ( it == loader->nodeByIdMap_.end() ) ? false : it->second->nodeType == NodeType::ColorGroup;
            } );

            if ( colorIt != refPids.end() )
            { 
                colorNode = loader->nodeByIdMap_[*colorIt];
                colorIndex = int( colorIt - refPids.begin() );
            }

            auto textureIt = std::find_if( refPids.begin(), refPids.end(), [&] ( int pid )
            {
                auto it = loader->nodeByIdMap_.find( pid );
                return ( it == loader->nodeByIdMap_.end() ) ? false : it->second->nodeType == NodeType::Texture2dGroup;
            } );

            if ( textureIt != refPids.end() )
            { 
                textureNode = loader->nodeByIdMap_[*textureIt];
                textureIndex = int( textureIt - refPids.begin() );
            }

            if ( colorNode && !loader->failedToLoadColoring )
            {
                for ( int i = 0; i < 3; ++i )
                {
                    if ( ps[i] < 0 || ps[i] >= it->second->pindices.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    if ( vs[i] < 0 || vs[i] >= vColorMap.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );

                    const auto& refPindices = it->second->pindices[ps[i]];

                    if ( refPindices[colorIndex] < 0 || refPindices[colorIndex] >= colorNode->colors.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );
                    vColorMap[VertId( vs[i] )] = colorNode->colors[refPindices[colorIndex]];
                }
            }

            if ( textureNode && !loader->failedToLoadColoring )
            {

                for ( int i = 0; i < 3; ++i )
                {
                    if ( ps[i] < 0 || ps[i] >= it->second->pindices.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    if ( vs[i] < 0 || vs[i] >= vUVCoords.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'v' attribute" ) );
                    const auto& refPindices = it->second->pindices[ps[i]];
                    if ( refPindices[textureIndex] < 0 || refPindices[textureIndex] >= textureNode->uvGroup.size() )
                        return unexpected( std::string( "3DF model triangle node has invalid 'p' attribute" ) );

                    auto& vertUV = vUVCoords[VertId( vs[i] )];
                    const auto refUV = textureNode->uvGroup[refPindices[textureIndex]];
                    // If vertex already has another UV coordinates, texture will be ignored
                    if ( !std::isnan( vertUV.x ) && ( vertUV.x != refUV.x || vertUV.y != refUV.y ) )
                    {
                        loader->failedToLoadColoring = true;
                        loader->warnings.append( "Texture cannot be show correctly because 3DF model has different UV coordinates for some vertices\n" );
                    }

                    vertUV = refUV;
                }
            }
        }        
    }

    if ( !reportProgress( callback, 0.5f ) )
        return unexpectedOperationCanceled();

    std::vector<MeshBuilder::VertDuplication> dups;
    int skippedFaceCount = 0;
    MeshBuilder::BuildSettings buildSettings{ .skippedFaceCount = &skippedFaceCount };

    if ( !fColorMap.empty() )
    {
        fColorMap.resize( tris.size(), loader->filamentColors[0] );
    }

    auto resMesh = std::make_shared<Mesh>( Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris, &dups, buildSettings ) );
    
    if ( !dups.empty() )
    {
        loader->duplicatedVertexCountAccum += int( dups.size() );
        if ( !vUVCoords.empty() )
        {
            vUVCoords.resize( resMesh->topology.lastValidVert() + 1 );
            for ( auto& dup : dups )
            {
                vUVCoords[dup.dupVert] = vUVCoords[dup.srcVert];
            }
        }

        if ( !vColorMap.empty() )
        {
            vColorMap.resize( resMesh->topology.lastValidVert() + 1 );
            for ( auto& dup : dups )
            {
                vColorMap[dup.dupVert] = vColorMap[dup.srcVert];
            }
        }
    }
    
    loader->skippedFaceCountAccum += skippedFaceCount;

    if ( !reportProgress( callback, 0.75f ) )
        return unexpectedOperationCanceled();

    auto objMesh = std::make_shared<ObjectMesh>();
    obj = objMesh;

    if ( !vColorMap.empty() )
    {
        objMesh->setVertsColorMap( std::move( vColorMap ) );
        objMesh->setColoringType( ColoringType::VertsColorMap );
    }
    else if ( !fColorMap.empty() )
    {
        objMesh->setFacesColorMap( std::move( fColorMap ) );
        objMesh->setColoringType( ColoringType::FacesColorMap );
    }

    if ( texId != -1 )
    {
        //if any vertex has NaN UV, we can't load the texture
        if ( std::none_of( begin( vUVCoords ), end( vUVCoords ), [] ( const auto& uv ) { return std::isnan( uv.x ); } ) )
        {
            auto it = loader->nodeByIdMap_.find( texId );
            if ( it == loader->nodeByIdMap_.end() )
                return unexpected( "Invalid texture id" );

            if ( it->second->texture.resolution.x > 0 && it->second->texture.resolution.y > 0 )
            {
                //cannot move because the same texture could be used in multiple objects
                objMesh->setTextures( { it->second->texture } );
                objMesh->setUVCoords( std::move( vUVCoords ) );
                objMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
            }
            else
            {
                loader->warnings.append( "Texture will not be loaded.\n" );
            }
        }
        else
        {
            loader->warnings.append( "Object " + obj->name() + " has incomplete UV coordinates. Texture will not be loaded.\n" );
        }
    }

    objMesh->setMesh( resMesh );

    if ( bgColorWasRead )
    {
        objMesh->setFrontColor( bgColor, true );
        objMesh->setFrontColor( bgColor, false );
    }
    return {};
}

void ThreeMFLoader::initFilamentColors_()
{
    if ( filamentInited_ )
        return;
    filamentInited_ = true;
    const Json::Value* foundProjSettings{ nullptr };
    for ( const auto& [path, json] : jsonDocuments_ )
    {
        if ( pathFromUtf8( path ).stem() == "project_settings" )
        {
            foundProjSettings = &json;
            break;
        }
    }
    if ( !foundProjSettings )
        return;
    const auto& projSettings = *foundProjSettings;
    if ( !projSettings["filament_colour"].isArray() )
        return;
    const auto& fColors = projSettings["filament_colour"];
    filamentColors.reserve( fColors.size() );
    for ( int i = 0; i < int( fColors.size() ); ++i )
    {
        if ( !fColors[i].isString() )
            continue;
        auto colorRes = parseColor( fColors[i].asString() );
        if ( !colorRes.has_value() )
            continue;
        filamentColors.emplace_back( std::move( *colorRes ) );
    }
    if ( filamentColors.size() != fColors.size() )
        filamentColors.clear();
}

void Node::setFilamentFaceColor_( FaceId f, const std::string& fId, FaceColors& fColorMap )
{
    if ( !loader->filamentInited_ )
        return;
    if ( loader->filamentColors.empty() )
        return;
    if ( fColorMap.size() <= f )
        fColorMap.resize( f + 1, loader->filamentColors[0] );

    // taken from https://github.com/bambulab/BambuStudio/issues/1892#issuecomment-1628513224
    constexpr std::array<const char*, 16> cIdMap =
    {
        "4", "8", "0C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "AC", "BC", "CC", "DC"
    };
    int foundFilId = -1;
    for ( int i = 0; i < cIdMap.size(); ++i )
    {
        if ( fId == cIdMap[i] )
        {
            foundFilId = i;
            break;
        }
    }
    if ( foundFilId != -1 && foundFilId < loader->filamentColors.size() )
        fColorMap[f] = loader->filamentColors[foundFilId];
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

Expected<LoadedObject> deserializeObjectTreeFrom3mf( const std::filesystem::path& path, const ProgressCallback& callback )
{
    const auto tmpFolder = UniqueTemporaryFolder( {} );

    auto resZip = decompressZip( path, tmpFolder );
    if ( !resZip )
        return unexpected( "ZIP container error: " + resZip.error() );

    if ( !reportProgress( callback, 0.1f ) )
        return unexpectedOperationCanceled();

    std::vector<std::filesystem::path> files;
    std::error_code ec;

    for ( auto const& dirEntry : DirectoryRecursive{ tmpFolder, ec } )
        if ( !dirEntry.is_directory( ec ) )
            files.push_back( dirEntry.path() );

    if ( files.empty() )
        return unexpected( "Could not find .model" );

    return ThreeMFLoader{}.load( files, tmpFolder, subprogress( callback, 0.1f, 0.9f ) );
}

Expected<LoadedObject> deserializeObjectTreeFromModel( const std::filesystem::path& path, const ProgressCallback& callback )
{
    return ThreeMFLoader{}.load( { path }, path.parent_path(), callback );
}

MR_ADD_SCENE_LOADER( IOFilter( "3D Manufacturing format (.3mf)", "*.3mf" ), deserializeObjectTreeFrom3mf )
MR_ADD_SCENE_LOADER( IOFilter( "3D Manufacturing model (.model)", "*.model" ), deserializeObjectTreeFromModel )

}
#endif
