#include "MRObjectMesh.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRMeshIntersect.h"
#include "MRStringConvert.h"
#include "MRAABBTree.h"
#include "MRAABBTreePoints.h"
#include "MRLine3.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRFmt.h"
#include "MRSceneSettings.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMesh )

MeshIntersectionResult ObjectMesh::worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region ) const
{
    MeshIntersectionResult res;
    if ( !mesh_ )
        return res;
    const AffineXf3f rayToMeshXf = worldXf().inverse();
    res = rayMeshIntersect( { *mesh_, region }, transformed( worldRay, rayToMeshXf ) );
    return res;
}

void ObjectMesh::setMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh == mesh_ )
        return;
    mesh_ = std::move(mesh);
    selectFaces({});
    selectEdges({});
    setCreases({});
    setDirtyFlags( DIRTY_ALL );
}

std::shared_ptr< Mesh > ObjectMesh::updateMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh != mesh_ )
    {
        mesh_.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
    }
    return mesh;
}

std::vector<std::string> ObjectMesh::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();

    if ( mesh_ )
    {
        res.push_back( "components: " + std::to_string( numComponents() ) );
        res.push_back( "handles: " + std::to_string( numHandles() ) );

        if ( mesh_->points.size() != mesh_->topology.vertSize() ||
             mesh_->points.capacity() != mesh_->topology.vertCapacity() )
        {
            res.push_back( "points: " + std::to_string( mesh_->points.size() ) + " size" );
            if ( mesh_->points.size() < mesh_->points.capacity() )
                res.back() += " / " + std::to_string( mesh_->points.capacity() ) + " capacity";
        }

        res.push_back( "vertices: " + std::to_string( mesh_->topology.numValidVerts() ) );
        if( mesh_->topology.numValidVerts() < mesh_->topology.vertSize() )
            res.back() += " / " + std::to_string( mesh_->topology.vertSize() ) + " size";
        if( mesh_->topology.vertSize() < mesh_->topology.vertCapacity() )
            res.back() += " / " + std::to_string( mesh_->topology.vertCapacity() ) + " capacity";

        res.push_back( "faces: " + std::to_string( mesh_->topology.numValidFaces() ) );
        const auto nFacesSelected = numSelectedFaces();
        if( nFacesSelected )
            res.back() += " / " + std::to_string( nFacesSelected ) + " selected";
        if( mesh_->topology.numValidFaces() < mesh_->topology.faceSize() )
            res.back() += " / " + std::to_string( mesh_->topology.faceSize() ) + " size";
        if( mesh_->topology.faceSize() < mesh_->topology.faceCapacity() )
            res.back() += " / " + std::to_string( mesh_->topology.faceCapacity() ) + " capacity";

        res.push_back( "edges: " + std::to_string( numUndirectedEdges() ) );
        if( auto nEdgesSelected = numSelectedEdges() )
            res.back() += " / " + std::to_string( nEdgesSelected ) + " selected";
        if( auto nCreaseEdges = numCreaseEdges() )
            res.back() += " / " + std::to_string( nCreaseEdges ) + " creases";
        if( numUndirectedEdges() < mesh_->topology.undirectedEdgeSize() )
            res.back() += " / " + std::to_string( mesh_->topology.undirectedEdgeSize() ) + " size";
        if( mesh_->topology.undirectedEdgeSize() < mesh_->topology.undirectedEdgeCapacity() )
            res.back() += " / " + std::to_string( mesh_->topology.undirectedEdgeCapacity() ) + " capacity";

        for ( TextureId i = TextureId{ 0 }; i < textures_.size(); ++i )
            res.push_back( "texture " + std::to_string( i ) + ": " + std::to_string( textures_[i].resolution.x) + " x " + std::to_string(textures_[i].resolution.y));

        if ( !uvCoordinates_.empty() )
        {
            res.push_back( "uv-coords: " + std::to_string( uvCoordinates_.size() ) );
            if ( uvCoordinates_.size() < uvCoordinates_.capacity() )
                res.back() += " / " + std::to_string( uvCoordinates_.capacity() ) + " capacity";
        }
        if ( !vertsColorMap_.empty() )
        {
            res.push_back( "colors: " + std::to_string( vertsColorMap_.size() ) );
            if ( vertsColorMap_.size() < vertsColorMap_.capacity() )
                res.back() += " / " + std::to_string( vertsColorMap_.capacity() ) + " capacity";
        }

        res.push_back( "holes: " + std::to_string( numHoles() ) );

        res.push_back( fmt::format( "area: {:.6}", totalArea() ) );
        if( nFacesSelected )
            res.back() += fmt::format( " / {:.6} selected", selectedArea() );
        if ( numHoles() == 0 )
            res.push_back( fmt::format( "volume: {:.6}", volume() ) );
        res.push_back( fmt::format( "avg edge len: {:.6}", avgEdgeLen() ) );

        boundingBoxToInfoLines_( res );

        size_t treesSize = 0;
        if ( auto tree = mesh_->getAABBTreeNotCreate() )
            treesSize += tree->heapBytes();
        if ( auto tree = mesh_->getAABBTreePointsNotCreate() )
            treesSize += tree->heapBytes();
        if ( treesSize > 0 )
            res.push_back( "AABB trees: " + bytesString( treesSize ) );
    }
    else
        res.push_back( "no mesh" );
    return res;
}

std::shared_ptr<Object> ObjectMesh::clone() const
{
    auto res = std::make_shared<ObjectMesh>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

std::shared_ptr<Object> ObjectMesh::shallowClone() const
{
    auto res = std::make_shared<ObjectMesh>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

void ObjectMesh::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectMeshHolder::setDirtyFlags( mask, invalidateCaches );
    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE)
    {
        if ( mesh_ )
        {
            meshChangedSignal( mask );
        }
    }
}

void ObjectMesh::swapBase_( Object& other )
{
    if ( auto otherMesh = other.asType<ObjectMesh>() )
        std::swap( *this, *otherMesh );
    else
        assert( false );
}

void ObjectMesh::swapSignals_( Object& other )
{
    ObjectMeshHolder::swapSignals_( other );
    if ( auto otherMesh = other.asType<ObjectMesh>() )
        std::swap( meshChangedSignal, otherMesh->meshChangedSignal );
    else
        assert( false );
}

void ObjectMesh::serializeFields_( Json::Value& root ) const
{
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( ObjectMesh::TypeName() );
}

std::shared_ptr<ObjectMesh> merge( const std::vector<std::shared_ptr<ObjectMesh>>& objsMesh )
{
    MR_TIMER

    bool hasVertColorMap = false;
    bool hasFaceColorMap = false;
    bool needSaveTexture = true;
    Vector2i resolution( -1, -1 );
    size_t numTexture = 0;
    size_t totalVerts = 0;
    size_t totalFaces = 0;
    size_t numObject = 0;
    for ( const auto& obj : objsMesh )
    {
        if ( auto curMesh = obj->mesh() )
        {
            totalVerts += curMesh->topology.numValidVerts();
            totalFaces += curMesh->topology.numValidFaces();
            if ( !obj->getVertsColorMap().empty() )
                hasVertColorMap = true;
            if ( !obj->getFacesColorMap().empty() )
                hasFaceColorMap = true;

            const auto& textures = obj->getTextures();
            if ( needSaveTexture )
            {
                if ( textures.empty() )
                {
                    needSaveTexture = false;
                    continue;
                }

                numTexture += textures.size();
                numObject++;
                
                if ( resolution.x == -1 )
                    resolution = textures.back().resolution;
                else if ( resolution != textures.back().resolution )
                    needSaveTexture = false;
            }
        }
    }
    auto mesh = std::make_shared<Mesh>();
    auto& points = mesh->points;
    mesh->topology.vertReserve( totalVerts );
    mesh->topology.faceReserve( totalFaces );

    VertColors vertColors;
    if ( hasVertColorMap )
        vertColors.resizeNoInit( totalVerts );

    FaceColors faceColors;
    if ( hasFaceColorMap )
        faceColors.resizeNoInit( totalFaces );

    VertUVCoords uvCoords;
    Vector<TextureId, FaceId> texturePerFace;
    Vector<MeshTexture, TextureId> textures;
    if ( needSaveTexture )
    {
        texturePerFace.resizeNoInit( totalFaces );
        uvCoords.resizeNoInit( totalVerts );
        textures.reserve( numTexture );
    }

    numObject = 0;
    TextureId previousNumTexture(-1);
    TextureId curNumTexture(-1);
    for ( const auto& obj : objsMesh )
    {
        if ( !obj->mesh() )
            continue;

        VertMap vertMap;
        FaceMap faceMap;
        mesh->addPart( *obj->mesh(), hasFaceColorMap || needSaveTexture ? &faceMap : nullptr, &vertMap );

        auto worldXf = obj->worldXf();
        for ( const auto& vInd : vertMap )
        {
            if ( vInd.valid() )
                points[vInd] = worldXf( points[vInd] );
        }

        if ( hasVertColorMap )
        {
            const auto& curColorMap = obj->getVertsColorMap();
            for ( VertId thisId = 0_v; thisId < vertMap.size(); ++thisId )
            {
                if ( auto mergeId = vertMap[thisId] )
                    vertColors[mergeId] = curColorMap.size() <= thisId ? obj->getFrontColor() : curColorMap[thisId];
            }
        }
        if ( hasFaceColorMap )
        {
            const auto& curColorMap = obj->getFacesColorMap();
            for ( FaceId thisId = 0_f; thisId < faceMap.size(); ++thisId )
            {
                if ( auto mergeId = faceMap[thisId] )
                    faceColors[mergeId] = curColorMap.size() <= thisId ? obj->getFrontColor() : curColorMap[thisId];
            }
        }
        if ( needSaveTexture )
        {
            const auto& curUvCoords = obj->getUVCoords();
            for ( VertId thisId = 0_v; thisId < vertMap.size(); ++thisId )
            {
                if ( auto mergeId = vertMap[thisId] )
                    uvCoords[mergeId] = curUvCoords[thisId];
            }
            const auto curTextures = obj->getTextures();
            textures.vec_.insert( textures.vec_.end(), curTextures.vec_.begin(), curTextures.vec_.end() );
            
            if ( numObject == 0 )
                curNumTexture = TextureId( 0 );
            else
                curNumTexture += previousNumTexture;

            previousNumTexture = TextureId( curTextures.size() );

            const auto& curTexturePerFace = obj->getTexturePerFace();
            const bool emptyTexturePerFace = curTexturePerFace.empty();
            for ( FaceId thisId = 0_f; thisId < faceMap.size(); ++thisId )
            {
                if ( auto mergeId = faceMap[thisId] )
                {
                    if ( emptyTexturePerFace )
                        texturePerFace[mergeId] = curNumTexture;
                    else
                        texturePerFace[mergeId] = curNumTexture + curTexturePerFace[thisId];
                }
            }
        }
        numObject++;
    }
    assert( totalVerts == mesh->topology.numValidVerts() );
    assert( totalFaces == mesh->topology.numValidFaces() );

    auto objectMesh = std::make_shared<ObjectMesh>();
    // if this data does not need to be set, it will be empty at this point
    objectMesh->setVertsColorMap( std::move( vertColors ) );
    objectMesh->setFacesColorMap( std::move( faceColors ) );
    objectMesh->setMesh( std::move( mesh ) );
    objectMesh->setTexturePerFace( std::move( texturePerFace ) );
    objectMesh->setTextures( std::move( textures ) );
    objectMesh->setUVCoords( std::move( uvCoords ) );
    if( hasVertColorMap )
        objectMesh->setColoringType( ColoringType::VertsColorMap );
    else if( hasFaceColorMap )
        objectMesh->setColoringType( ColoringType::FacesColorMap );

    ViewportMask flat = ViewportMask::all();
    ViewportMask smooth = ViewportMask::all();
    for ( const auto& obj : objsMesh )
    {
        ViewportMask shading = obj->getVisualizePropertyMask( MeshVisualizePropertyType::FlatShading );
        flat &= shading;
        smooth &= ( ~shading );
    }

    if ( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat )
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::FlatShading, ~smooth );
    else
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::FlatShading, flat );

    if ( !objectMesh->getTextures().empty() )
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::Texture, ViewportMask::all() );

    return objectMesh;
}

TEST(MRMesh, DataModel)
{
    Object root;
    EXPECT_EQ(root.children().size(), 0);

    auto child = std::make_shared<Object>();
    EXPECT_TRUE(root.addChild(child));
    EXPECT_FALSE(root.addChild(child));
    EXPECT_EQ(&root, child->parent());
    EXPECT_EQ(root.children().size(), 1);

    child->setName( "child" );
    EXPECT_EQ( child, root.find( "child" ) );
    EXPECT_FALSE( root.find( "something" ) );
    EXPECT_EQ( child, root.find<Object>( "child" ) );
    EXPECT_FALSE( root.find<ObjectMesh>( "child" ) );

    auto grandchild = std::make_shared<ObjectMesh>();
    EXPECT_TRUE(child->addChild(grandchild));
    EXPECT_EQ(child.get(), grandchild->parent());

    EXPECT_TRUE(root.removeChild(child));
    EXPECT_FALSE(root.removeChild(child));
    EXPECT_EQ(nullptr, child->parent());
    EXPECT_EQ(root.children().size(), 0);

    child->removeAllChildren();
    EXPECT_EQ(child->children().size(), 0);
}

} //namespace MR
