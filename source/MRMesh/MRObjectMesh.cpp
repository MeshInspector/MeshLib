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
#include "MRSceneSettings.h"
#include "MRMapOrHashMap.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRFmt.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMesh )

MeshIntersectionResult ObjectMesh::worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region ) const
{
    MeshIntersectionResult res;
    if ( !data_.mesh )
        return res;
    const AffineXf3f rayToMeshXf = worldXf().inverse();
    res = rayMeshIntersect( { *data_.mesh, region }, transformed( worldRay, rayToMeshXf ) );
    return res;
}

void ObjectMesh::setMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh == data_.mesh )
        return;
    data_.mesh = std::move(mesh);
    selectFaces({});
    selectEdges({});
    setCreases({});
    setDirtyFlags( DIRTY_ALL );
}

std::shared_ptr< Mesh > ObjectMesh::updateMesh( std::shared_ptr< Mesh > mesh )
{
    if ( mesh != data_.mesh )
    {
        data_.mesh.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
    }
    return mesh;
}

std::vector<std::string> ObjectMesh::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();

    if ( data_.mesh )
    {
        res.push_back( "components: " + std::to_string( numComponents() ) );
        res.push_back( "handles: " + std::to_string( numHandles() ) );

        if ( data_.mesh->points.size() != data_.mesh->topology.vertSize() ||
             data_.mesh->points.capacity() != data_.mesh->topology.vertCapacity() )
        {
            res.push_back( "points: " + std::to_string( data_.mesh->points.size() ) + " size" );
            if ( data_.mesh->points.size() < data_.mesh->points.capacity() )
                res.back() += " / " + std::to_string( data_.mesh->points.capacity() ) + " capacity";
        }

        res.push_back( "vertices: " + std::to_string( data_.mesh->topology.numValidVerts() ) );
        if( data_.mesh->topology.numValidVerts() < data_.mesh->topology.vertSize() )
            res.back() += " / " + std::to_string( data_.mesh->topology.vertSize() ) + " size";
        if( data_.mesh->topology.vertSize() < data_.mesh->topology.vertCapacity() )
            res.back() += " / " + std::to_string( data_.mesh->topology.vertCapacity() ) + " capacity";

        res.push_back( "triangles: " + std::to_string( data_.mesh->topology.numValidFaces() ) );
        const auto nFacesSelected = numSelectedFaces();
        if( nFacesSelected )
            res.back() += " / " + std::to_string( nFacesSelected ) + " selected";
        if( data_.mesh->topology.numValidFaces() < data_.mesh->topology.faceSize() )
            res.back() += " / " + std::to_string( data_.mesh->topology.faceSize() ) + " size";
        if( data_.mesh->topology.faceSize() < data_.mesh->topology.faceCapacity() )
            res.back() += " / " + std::to_string( data_.mesh->topology.faceCapacity() ) + " capacity";

        res.push_back( "edges: " + std::to_string( numUndirectedEdges() ) );
        if( auto nEdgesSelected = numSelectedEdges() )
            res.back() += " / " + std::to_string( nEdgesSelected ) + " selected";
        if( auto nCreaseEdges = numCreaseEdges() )
            res.back() += " / " + std::to_string( nCreaseEdges ) + " creases";
        if( numUndirectedEdges() < data_.mesh->topology.undirectedEdgeSize() )
            res.back() += " / " + std::to_string( data_.mesh->topology.undirectedEdgeSize() ) + " size";
        if( data_.mesh->topology.undirectedEdgeSize() < data_.mesh->topology.undirectedEdgeCapacity() )
            res.back() += " / " + std::to_string( data_.mesh->topology.undirectedEdgeCapacity() ) + " capacity";

        for ( TextureId i = TextureId{ 0 }; i < textures_.size(); ++i )
            res.push_back( "texture " + std::to_string( i ) + ": " + std::to_string( textures_[i].resolution.x) + " x " + std::to_string(textures_[i].resolution.y));

        res.push_back( std::string( "coloring type: " ) + asString( getColoringType() ) );

        if ( !data_.uvCoordinates.empty() )
        {
            res.push_back( "uv-coords: " + std::to_string( data_.uvCoordinates.size() ) );
            if ( data_.uvCoordinates.size() < data_.uvCoordinates.capacity() )
                res.back() += " / " + std::to_string( data_.uvCoordinates.capacity() ) + " capacity";
        }
        if ( !data_.vertColors.empty() )
        {
            res.push_back( "colors: " + std::to_string( data_.vertColors.size() ) );
            if ( data_.vertColors.size() < data_.vertColors.capacity() )
                res.back() += " / " + std::to_string( data_.vertColors.capacity() ) + " capacity";
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
        if ( auto tree = data_.mesh->getAABBTreeNotCreate() )
            treesSize += tree->heapBytes();
        if ( auto tree = data_.mesh->getAABBTreePointsNotCreate() )
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
    if ( data_.mesh )
        res->data_.mesh = std::make_shared<Mesh>( *data_.mesh );
    return res;
}

std::shared_ptr<Object> ObjectMesh::shallowClone() const
{
    auto res = std::make_shared<ObjectMesh>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = data_.mesh;
    return res;
}

void ObjectMesh::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectMeshHolder::setDirtyFlags( mask, invalidateCaches );
    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE)
    {
        if ( data_.mesh )
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
    MR_TIMER;

        bool hasVertColorMap = false; // save if at least one has
    bool hasFaceColorMap = false;     // save if at least one has
    bool needSaveUVCoords = true;     // save if all have

    bool needSaveTextures = true;     // save if all have textures of same size
    bool exactSameTextures = true;     // true if all object has identical textures

    const Vector<MeshTexture, TextureId>* prevObjTextures = nullptr;


    size_t numTextures = 0;
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
            if ( obj->getUVCoords().empty() )
                needSaveUVCoords = false;

            if ( !needSaveTextures )
                continue;

            const auto& textures = obj->getTextures();
            numTextures += textures.size();
            if ( textures.empty() )
            {
                needSaveTextures = false;
                continue;
            }
            if ( !prevObjTextures )
            {
                prevObjTextures = &textures;
                continue;
            }
            assert( prevObjTextures );
            if ( prevObjTextures->size() != textures.size() )
                exactSameTextures = false;
            if ( prevObjTextures->front().resolution != textures.front().resolution )
            {
                needSaveTextures = false;
                continue;
            }
            if ( !exactSameTextures )
                continue;
            for ( int i = 0; i < prevObjTextures->size(); ++i )
            {
                if ( ( *prevObjTextures )[TextureId( i )].pixels != textures[TextureId( i )].pixels )
                {
                    exactSameTextures = false;
                    break;
                }
            }
        }
    }
    bool needTexturePerFace = needSaveTextures && ( !exactSameTextures || ( prevObjTextures && prevObjTextures->size() > 1 ) );

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
    if ( needSaveUVCoords )
        uvCoords.resizeNoInit( totalVerts );

    TexturePerFace texturePerFace;
    if ( needTexturePerFace )
        texturePerFace.resizeNoInit( totalFaces );

    Vector<MeshTexture, TextureId> mergedTextures;
    if ( needSaveTextures && !exactSameTextures )
        mergedTextures.reserve( numTextures );

    numObject = 0;
    TextureId previousNumTexture(-1);
    TextureId curNumTexture(-1);
    for ( const auto& obj : objsMesh )
    {
        if ( !obj->mesh() )
            continue;

        VertMap vertMap;
        FaceMap faceMap;
        mesh->addMesh( *obj->mesh(), hasFaceColorMap || needTexturePerFace ? &faceMap : nullptr, &vertMap );

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
        if ( needSaveUVCoords )
        {
            const auto& curUvCoords = obj->getUVCoords();
            for ( VertId thisId = 0_v; thisId < vertMap.size(); ++thisId )
            {
                if ( auto mergeId = vertMap[thisId] )
                    uvCoords[mergeId] = curUvCoords[thisId];
            }
        }

        if ( needSaveTextures )
        {
            if ( exactSameTextures )
            {
                if ( needTexturePerFace )
                {
                    const auto& curTextPerFace = obj->getTexturePerFace();
                    for ( FaceId thisId = 0_f; thisId < faceMap.size(); ++thisId )
                    {
                        if ( auto mergeId = faceMap[thisId] )
                            texturePerFace[mergeId] = curTextPerFace[thisId];
                    }
                }
            }
            else
            {
                const auto& curTextures = obj->getTextures();
                mergedTextures.vec_.insert( mergedTextures.vec_.end(), curTextures.vec_.begin(), curTextures.vec_.end() );

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
    if ( needSaveTextures )
    {
        if ( exactSameTextures && prevObjTextures )
            objectMesh->setTextures( *prevObjTextures );
        else
            objectMesh->setTextures( std::move( mergedTextures ) );
    }
    objectMesh->setUVCoords( std::move( uvCoords ) );
    if( hasVertColorMap )
        objectMesh->setColoringType( ColoringType::VertsColorMap );
    else if( hasFaceColorMap )
        objectMesh->setColoringType( ColoringType::FacesColorMap );

    ViewportMask flat = ViewportMask::all();
    ViewportMask smooth = ViewportMask::all();
    ViewportMask textures = ViewportMask::all();
    for ( const auto& obj : objsMesh )
    {
        ViewportMask shading = obj->getVisualizePropertyMask( MeshVisualizePropertyType::FlatShading );
        flat &= shading;
        smooth &= ( ~shading );
        textures &= obj->getVisualizePropertyMask( MeshVisualizePropertyType::Texture );
    }

    if ( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat )
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::FlatShading, ~smooth );
    else
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::FlatShading, flat );

    if ( !objectMesh->getTextures().empty() )
        objectMesh->setVisualizePropertyMask( MeshVisualizePropertyType::Texture, textures );

    return objectMesh;
}

std::shared_ptr<MR::ObjectMesh> cloneRegion( const std::shared_ptr<ObjectMesh>& objMesh, const FaceBitSet& region, bool copyTexture /*= true */ )
{
    VertMapOrHashMap vertMap;
    FaceMapOrHashMap faceMap;
    PartMapping partMapping;
    if ( !objMesh->getVertsColorMap().empty() || !objMesh->getUVCoords().empty() )
        partMapping.tgt2srcVerts = &vertMap;
    if ( !objMesh->getFacesColorMap().empty() || !objMesh->getTexturePerFace().empty() )
        partMapping.tgt2srcFaces = &faceMap;
    std::shared_ptr<Mesh> newMesh = std::make_shared<Mesh>( objMesh->mesh()->cloneRegion( region, false, partMapping ) );
    std::shared_ptr<ObjectMesh> newObj = std::make_shared<ObjectMesh>();
    newObj->setFrontColor( objMesh->getFrontColor( true ), true );
    newObj->setFrontColor( objMesh->getFrontColor( false ), false );
    newObj->setBackColor( objMesh->getBackColor() );
    newObj->setMesh( newMesh );
    newObj->setAllVisualizeProperties( objMesh->getAllVisualizeProperties() );
    if ( copyTexture )
    {
        newObj->copyTextureAndColors( *objMesh, *vertMap.getMap(), *faceMap.getMap() );
    }
    else
    {
        newObj->copyColors( *objMesh, *vertMap.getMap(), *faceMap.getMap() );
        newObj->setVisualizePropertyMask( MeshVisualizePropertyType::Texture, ViewportMask( 0 ) );
    }
    newObj->setName( objMesh->name() + "_part" );
    return newObj;
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
