#include "MRObjectMeshHolder.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRSerializer.h"
#include "MRMeshLoad.h"
#include "MRSceneColors.h"
#include "MRIRenderObject.h"
#include "MRViewportId.h"
#include "MRSceneSettings.h"
#include "MRHeapBytes.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRDirectory.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRAsyncLaunchType.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMeshHolder )

const Color& ObjectMeshHolder::getSelectedFacesColor( ViewportId id ) const
{
    return faceSelectionColor_.get( id );
}

const Color& ObjectMeshHolder::getSelectedEdgesColor( ViewportId id ) const
{
    return edgeSelectionColor_.get( id );
}

void ObjectMeshHolder::setSelectedFacesColor( const Color& color, ViewportId id )
{
    if ( color == faceSelectionColor_.get( id ) )
        return;
    faceSelectionColor_.set( color, id );
    needRedraw_ = true;
}

void ObjectMeshHolder::setSelectedEdgesColor( const Color& color, ViewportId id )
{
    if ( color == edgeSelectionColor_.get( id ) )
        return;
    edgeSelectionColor_.set( color, id );
    needRedraw_ = true;
}

Expected<std::future<Expected<void>>> ObjectMeshHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !data_.mesh )
        return {};

    SaveSettings saveSettings;
    saveSettings.onlyValidPoints = false;
    saveSettings.packPrimitives = false;
    if ( !data_.vertColors.empty() )
        saveSettings.colors = &data_.vertColors;
    auto save = [mesh = data_.mesh, serializeFormat = std::string( actualSerializeFormat() ), path, saveSettings]() -> Expected<void>
    {
        const auto extension = std::string( "*" ) + serializeFormat;
        auto meshSaver = MeshSave::getMeshSaver( extension );
        if ( meshSaver.fileSave )
        {
            auto filename = path;
            filename += serializeFormat;
            return meshSaver.fileSave( *mesh, filename, saveSettings );
        }
        return unexpectedUnsupportedFileExtension();
    };
    return std::async( getAsyncLaunchType(), save );
}

void ObjectMeshHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowTexture"] = showTexture_.value();
    root["ShowFaces"] = showFaces_.value();
    root["ShowLines"] = showEdges_.value();
    root["ShowPoints"] = showPoints_.value();
    root["ShowBordersHighlight"] = showBordersHighlight_.value();
    root["ShowSelectedEdges"] = showSelectedEdges_.value();
    root["ShowSelectedFaces"] = showSelectedFaces_.value();
    root["OnlyOddFragments"] = onlyOddFragments_.value();
    root["PolygonOffset"] = polygonOffset_.value();
    root["ShadingEnabled"] = shadingEnabled_.value();
    root["FaceBased"] = flatShading_.contains( MR::ViewportId( 1 ) );
    switch( coloringType_ )
    {
    case ColoringType::VertsColorMap:
        root["ColoringType"] = "PerVertex";
        break;
    case ColoringType::FacesColorMap:
        root["ColoringType"] = "PerFace";
        break;
    default:
        root["ColoringType"] = "Solid";
    }
    serializeToJson( data_.faceColors.vec_, root["FaceColors"] );

    const auto meshSaver = MeshSave::getMeshSaver( std::string( "*" ) + actualSerializeFormat() );
    if ( !meshSaver.capabilities.storesVertexColors )
        serializeToJson( data_.vertColors.vec_, root["VertColors"] );

    // texture
    if ( !textures_.empty() )
    {
        root["TextureCount"] = int ( textures_.size() );
        for ( TextureId i = TextureId{ 0 }; i < textures_.size(); ++i )
            serializeToJson( textures_[i], root["Textures"][std::to_string( i )] );
    }

    // texture id per face id
    serializeToJson( data_.texturePerFace.vec_, root["TexturePerFace"] );

    serializeToJson( data_.uvCoordinates.vec_, root["UVCoordinates"] );
    // edges
    serializeToJson( Vector4f( edgesColor_.get() ), root["Colors"]["Edges"] );
    // vertices
    serializeToJson( Vector4f( pointsColor_.get() ), root["Colors"]["Points"] );
    // borders
    serializeToJson( Vector4f( bordersColor_.get() ), root["Colors"]["Borders"] );

    serializeToJson( Vector4f( faceSelectionColor_.get() ), root["Colors"]["Selection"]["Diffuse"] );

    serializeToJson( data_.selectedFaces, root["SelectionFaceBitSet"] );
    if ( data_.mesh )
    {
        serializeViaVerticesToJson( data_.selectedEdges, data_.mesh->topology, root["SelectionEdgeBitSet"] );
        serializeViaVerticesToJson( data_.creases, data_.mesh->topology, root["MeshCreasesUndirEdgeBitSet"] );
    }
    else
    {
        serializeToJson( data_.selectedEdges, root["SelectionEdgeBitSet"] );
        serializeToJson( data_.creases, root["MeshCreasesUndirEdgeBitSet"] );
    }

    root["PointSize"] = pointSize_;

    root["Type"].append( ObjectMeshHolder::TypeName() );
}

void ObjectMeshHolder::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    const auto& selectionColor = root["Colors"]["Selection"];

    if ( root["ShowTexture"].isUInt() )
        showTexture_ = ViewportMask{ root["ShowTexture"].asUInt() };
    if ( root["ShowFaces"].isUInt() )
        showFaces_ = ViewportMask{ root["ShowFaces"].asUInt() };
    if ( root["ShowLines"].isUInt() )
        showEdges_ = ViewportMask{ root["ShowLines"].asUInt() };
    if ( root["ShowPoints"].isUInt() )
        showPoints_ = ViewportMask{ root["ShowPoints"].asUInt() };
    if ( root["ShowBordersHighlight"].isUInt() )
        showBordersHighlight_ = ViewportMask{ root["ShowBordersHighlight"].asUInt() };
    if ( root["ShowSelectedEdges"].isUInt() )
        showSelectedEdges_ = ViewportMask{ root["ShowSelectedEdges"].asUInt() };
    if ( root["ShowSelectedFaces"].isUInt() )
        showSelectedFaces_ = ViewportMask{ root["ShowSelectedFaces"].asUInt() };
    if ( root["OnlyOddFragments"].isUInt() )
        onlyOddFragments_ = ViewportMask{ root["OnlyOddFragments"].asUInt() };
    if ( root["PolygonOffset"].isUInt() )
        polygonOffset_ = ViewportMask{ root["PolygonOffset"].asUInt() };
    if ( root["ShadingEnabled"].isUInt() )
        shadingEnabled_ = ViewportMask{ root["ShadingEnabled"].asUInt() };
    if ( root["FaceBased"].isBool() ) // Support old versions
        flatShading_ = root["FaceBased"].asBool() ? ViewportMask::all() : ViewportMask{};
    if ( root["ColoringType"].isString() )
    {
        const auto stype = root["ColoringType"].asString();
        if ( stype == "PerVertex" )
            setColoringType( ColoringType::VertsColorMap );
        else if ( stype == "PerFace" )
            setColoringType( ColoringType::FacesColorMap );
    }
    deserializeFromJson( root["FaceColors"], data_.faceColors.vec_ );
    deserializeFromJson( root["VertColors"], data_.vertColors.vec_ );

    Vector4f resVec;
    deserializeFromJson( selectionColor["Diffuse"], resVec );
    faceSelectionColor_.set( Color( resVec ) );
    // texture
    TextureId textureCount = TextureId{ 0 };
    if ( root["TextureCount"].isUInt() )
    {
        textureCount = TextureId{ root["TextureCount"].asInt() };
        textures_.resize( textureCount );

        for ( TextureId textureIndex = TextureId{ 0 }; textureIndex < textureCount; ++textureIndex )
            deserializeFromJson( root["Textures"][std::to_string( textureIndex )], textures_[textureIndex] );
    }
    else if ( root["Texture"].isObject() )
    {
        if ( textures_.empty() )
            textures_.resize( 1 );

        deserializeFromJson( root["Texture"], textures_.front() );
    }

    if ( root["TexturePerFace"].isObject() )
        deserializeFromJson( root["TexturePerFace"], data_.texturePerFace.vec_ );

    if ( root["UVCoordinates"].isObject() )
        deserializeFromJson( root["UVCoordinates"], data_.uvCoordinates.vec_ );
    // edges
    deserializeFromJson( root["Colors"]["Edges"], resVec );
    edgesColor_.set( Color( resVec ) );
    // vertices
    deserializeFromJson( root["Colors"]["Points"], resVec );
    pointsColor_.set( Color( resVec ) );
    // borders
    deserializeFromJson( root["Colors"]["Borders"], resVec );
    bordersColor_.set( Color( resVec ) );

    deserializeFromJson( root["SelectionFaceBitSet"], data_.selectedFaces );

    if ( data_.mesh )
    {
        data_.selectedFaces &= data_.mesh->topology.getValidFaces();

        const auto notLoneEdges = data_.mesh->topology.findNotLoneUndirectedEdges();
        deserializeViaVerticesFromJson( root["SelectionEdgeBitSet"], data_.selectedEdges, data_.mesh->topology );
        data_.selectedEdges &= notLoneEdges;

        deserializeViaVerticesFromJson( root["MeshCreasesUndirEdgeBitSet"], data_.creases, data_.mesh->topology );
        data_.creases &= notLoneEdges;
    }
    else
    {
        deserializeFromJson( root["SelectionEdgeBitSet"], data_.selectedEdges );
        deserializeFromJson( root["MeshCreasesUndirEdgeBitSet"], data_.creases );
    }

    if ( const auto& pointSizeJson = root["PointSize"]; pointSizeJson.isDouble() )
        pointSize_ = float( pointSizeJson.asDouble() );

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();
}

Expected<void> ObjectMeshHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    data_.vertColors.clear();
    auto modelPath = pathFromUtf8( utf8string( path ) + ".ctm" ); //quick path for most used format
    std::error_code ec;
    if ( !is_regular_file( modelPath, ec ) )
    {
        modelPath = findPathWithExtension( path );
        if ( modelPath.empty() )
            return unexpected( "No mesh file found: " + utf8string( path ) );
    }
    auto res = MeshLoad::fromAnySupportedFormat( modelPath, { .colors = &data_.vertColors, .callback = progressCb } );
    if ( !res.has_value() )
        return unexpected( res.error() );

    data_.mesh = std::make_shared<Mesh>( std::move( res.value() ) );
    return {};
}

Box3f ObjectMeshHolder::computeBoundingBox_() const
{
    if ( !data_.mesh )
        return Box3f();
    return data_.mesh->computeBoundingBox();
}

bool ObjectMeshHolder::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return VisualObject::supportsVisualizeProperty( type ) || type.tryGet<MeshVisualizePropertyType>().has_value();
}

AllVisualizeProperties ObjectMeshHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<MeshVisualizePropertyType>( ret );
    return ret;
}

void ObjectMeshHolder::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<MeshVisualizePropertyType>( properties, pos );
}

const ViewportMask &ObjectMeshHolder::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<MeshVisualizePropertyType>() )
    {
        switch ( *value )
        {
            case MeshVisualizePropertyType::Faces:
                return showFaces_;
            case MeshVisualizePropertyType::Texture:
                return showTexture_;
            case MeshVisualizePropertyType::Edges:
                return showEdges_;
            case MeshVisualizePropertyType::Points:
                return showPoints_;
            case MeshVisualizePropertyType::FlatShading:
                return flatShading_;
            case MeshVisualizePropertyType::EnableShading:
                return shadingEnabled_;
            case MeshVisualizePropertyType::OnlyOddFragments:
                return onlyOddFragments_;
            case MeshVisualizePropertyType::BordersHighlight:
                return showBordersHighlight_;
            case MeshVisualizePropertyType::SelectedEdges:
                return showSelectedEdges_;
            case MeshVisualizePropertyType::SelectedFaces:
                return showSelectedFaces_;
            case MeshVisualizePropertyType::PolygonOffsetFromCamera:
                return polygonOffset_;
            case MeshVisualizePropertyType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectMeshHolder::setEdgeWidth( float edgeWidth )
{
    if ( edgeWidth_ == edgeWidth )
        return;

    edgeWidth_ = edgeWidth;
    needRedraw_ = true;
}

void ObjectMeshHolder::setPointSize( float size )
{
    if ( pointSize_ == size )
        return;

    pointSize_ = size;
    needRedraw_ = true;
}

void ObjectMeshHolder::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectMeshHolder>( *this );
}

void ObjectMeshHolder::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setSelectedFacesColor( SceneColors::get( SceneColors::SelectedFaces ) );
    setSelectedEdgesColor( SceneColors::get( SceneColors::SelectedEdges ) );
    setEdgesColor( SceneColors::get( SceneColors::Edges ) );
    setPointsColor( SceneColors::get( SceneColors::Points ) );
    setBordersColor( SceneColors::get( SceneColors::Labels ) );
}

ObjectMeshHolder::ObjectMeshHolder()
{
    setDefaultSceneProperties_();
}


// for backward compatibility
const MeshTexture& ObjectMeshHolder::getTexture() const
{
    static const MeshTexture defaultTexture;
    return textures_.size() ? textures_.front() : defaultTexture;
}

void ObjectMeshHolder::setTexture( MeshTexture texture )
{
    if ( textures_.empty() )
        textures_.push_back( std::move( texture ) );
    else
        textures_.front() = std::move(texture);

    setDirtyFlags( DIRTY_TEXTURE );
}

void ObjectMeshHolder::updateTexture( MeshTexture& updated )
{
    if ( textures_.empty() )
        textures_.resize( 1 );

    std::swap( textures_.front(), updated);
    setDirtyFlags( DIRTY_TEXTURE );
}

void ObjectMeshHolder::copyTextureAndColors( const ObjectMeshHolder & src, const VertMap & thisToSrc, const FaceMap & thisToSrcFaces )
{
    MR_TIMER;
    copyColors( src, thisToSrc, thisToSrcFaces );
    setTextures( src.getTextures() );

    const auto& srcTexPerFace = src.getTexturePerFace();
    if ( !srcTexPerFace.empty() )
    {
        TexturePerFace texPerFace;
        texPerFace.resizeNoInit( thisToSrcFaces.size() );
        ParallelFor( texPerFace, [&] ( FaceId id )
        {
            texPerFace[id] = srcTexPerFace[thisToSrcFaces[id]];
        } );
        setTexturePerFace( std::move( texPerFace ) );
    }

    const auto& srcUVCoords = src.getUVCoords();
    const auto lastVert = src.mesh()->topology.lastValidVert();
    const bool updateUV = lastVert < srcUVCoords.size();

    if ( !updateUV )
        return;

    VertUVCoords uvCoords;
    uvCoords.resizeNoInit( thisToSrc.size() );
    ParallelFor( uvCoords, [&]( VertId id )
    {
        uvCoords[id] = srcUVCoords[thisToSrc[id]];
    } );

    setUVCoords( std::move( uvCoords ) );
}

void ObjectMeshHolder::copyColors( const ObjectMeshHolder& src, const VertMap& thisToSrc, const FaceMap& thisToSrcFaces )
{
    MR_TIMER;

    setColoringType( src.getColoringType() );

    const auto& srcColorMap = src.getVertsColorMap();
    if ( srcColorMap.empty() )
        return;

    VertColors colorMap;
    colorMap.resizeNoInit( thisToSrc.size() );
    ParallelFor( colorMap, [&] ( VertId id )
    {
        auto curId = thisToSrc[id];
        if( curId.valid() )
            colorMap[id] = srcColorMap[curId];
    } );
    setVertsColorMap( std::move( colorMap ) );

    if ( !data_.faceColors.empty() && data_.mesh )
    {
        const auto& validFace = data_.mesh->topology.getValidFaces();
        FaceColors faceColors;
        faceColors.resizeNoInit( validFace.size() );

        Color color = data_.faceColors[thisToSrcFaces[validFace.backId()]];
        bool differentColor = false;

        for ( const auto& faceId : validFace )
        {
            if ( !thisToSrcFaces[faceId].valid() )
                continue;

            auto& newColor = data_.faceColors[thisToSrcFaces[faceId]];
            faceColors[faceId] = newColor;
            if ( color != newColor )
                differentColor = true;
        }

        if ( differentColor )
            setFacesColorMap( std::move( faceColors ) );
        else if ( src.getColoringType() == ColoringType::FacesColorMap )
        {
            setFrontColor( color, true );
            setColoringType( ColoringType::SolidColor );
        }
    }
}

void ObjectMeshHolder::clearAncillaryTexture()
{
    if ( !ancillaryTexture_.pixels.empty() )
        setAncillaryTexture( {} );
    if ( !ancillaryUVCoordinates_.empty() )
        setAncillaryUVCoords( {} );
}

uint32_t ObjectMeshHolder::getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const
{
    auto flatShading = getVisualizePropertyMask( MeshVisualizePropertyType::FlatShading );
    uint32_t res = 0;
    if ( !( flatShading & viewportMask ).empty() )
    {
        res |= ( getDirtyFlags() & DIRTY_FACES_RENDER_NORMAL );
    }
    if ( ( flatShading & viewportMask ) != viewportMask )
    {
        if ( !data_.creases.any() )
        {
            res |= ( getDirtyFlags() & DIRTY_VERTS_RENDER_NORMAL );
        }
        else
        {
            res |= ( getDirtyFlags() & DIRTY_CORNERS_RENDER_NORMAL );
        }
    }
    return res;
}

bool ObjectMeshHolder::getRedrawFlag( ViewportMask viewportMask ) const
{
    return Object::getRedrawFlag( viewportMask ) ||
        ( isVisible( viewportMask ) &&
          ( getDirtyFlags() & ( ~( DIRTY_CACHES | ( DIRTY_RENDER_NORMALS - getNeededNormalsRenderDirtyValue( viewportMask ) ) ) ) ) );
}

void ObjectMeshHolder::applyScale( float scaleFactor )
{
    if ( !data_.mesh )
        return;

    auto& points = data_.mesh->points;

    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )points.size() ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            points[VertId( i )] *= scaleFactor;
        }
    } );
    setDirtyFlags( DIRTY_POSITION );
}

bool ObjectMeshHolder::hasVisualRepresentation() const
{
    return data_.mesh && numUndirectedEdges() > 0;
}

std::shared_ptr<Object> ObjectMeshHolder::clone() const
{
    auto res = std::make_shared<ObjectMeshHolder>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = std::make_shared<Mesh>( *data_.mesh );
    return res;
}

std::shared_ptr<Object> ObjectMeshHolder::shallowClone() const
{
    auto res = std::make_shared<ObjectMeshHolder>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = data_.mesh;
    return res;
}

void ObjectMeshHolder::selectFaces( FaceBitSet newSelection )
{
    data_.selectedFaces = std::move( newSelection );
    numSelectedFaces_.reset();
    selectedArea_.reset();
    faceSelectionChangedSignal();
    setDirtyFlags( DIRTY_SELECTION );
}

void ObjectMeshHolder::selectEdges( UndirectedEdgeBitSet newSelection )
{
    data_.selectedEdges = std::move( newSelection );
    numSelectedEdges_.reset();
    edgeSelectionChangedSignal();
    setDirtyFlags( DIRTY_EDGES_SELECTION );
}

bool ObjectMeshHolder::isMeshClosed() const
{
    if ( !meshIsClosed_ )
        meshIsClosed_ = data_.mesh && data_.mesh->topology.isClosed();

    return *meshIsClosed_;
}

Box3f ObjectMeshHolder::getWorldBox( ViewportId id ) const
{
    if ( !data_.mesh )
        return {};
    bool isDef = true;
    const auto worldXf = this->worldXf( id, &isDef );
    if ( isDef )
        id = {};
    auto & cache = worldBox_[id];
    if ( auto v = cache.get( worldXf ) )
        return *v;
    const auto box = data_.mesh->computeBoundingBox( &worldXf );
    cache.set( worldXf, box );
    return box;
}

size_t ObjectMeshHolder::numSelectedFaces() const
{
    if ( !numSelectedFaces_ )
    {
        numSelectedFaces_ = data_.selectedFaces.count();
#ifndef NDEBUG
        // check that there are no selected invalid faces
        assert( !data_.mesh || data_.selectedFaces.is_subset_of( data_.mesh->topology.getValidFaces() ) );
#endif
    }

    return *numSelectedFaces_;
}

size_t ObjectMeshHolder::numSelectedEdges() const
{
    if ( !numSelectedEdges_ )
    {
        numSelectedEdges_ = data_.selectedEdges.count();
#ifndef NDEBUG
        // check that there are no selected invalid edges
        assert( !data_.mesh || data_.selectedEdges.is_subset_of( data_.mesh->topology.findNotLoneUndirectedEdges() ) );
#endif
    }

    return *numSelectedEdges_;
}

size_t ObjectMeshHolder::numCreaseEdges() const
{
    if ( !numCreaseEdges_ )
    {
        numCreaseEdges_ = data_.creases.count();
#ifndef NDEBUG
        // check that there are no invalid edges among creases
        assert( !data_.mesh || data_.creases.is_subset_of( data_.mesh->topology.findNotLoneUndirectedEdges() ) );
#endif
    }

    return *numCreaseEdges_;
}

double ObjectMeshHolder::totalArea() const
{
    if ( !totalArea_ )
        totalArea_ = data_.mesh ? data_.mesh->area() : 0.0;

    return *totalArea_;
}

double ObjectMeshHolder::selectedArea() const
{
    if ( !selectedArea_ )
        selectedArea_ = data_.mesh ? data_.mesh->area( &data_.selectedFaces ) : 0.0;

    return *selectedArea_;
}

double ObjectMeshHolder::volume() const
{
    if ( !volume_ )
        volume_ = data_.mesh ? data_.mesh->volume() : 0.0;

    return *volume_;
}

float ObjectMeshHolder::avgEdgeLen() const
{
    if ( !avgEdgeLen_ )
        avgEdgeLen_ = data_.mesh ? data_.mesh->averageEdgeLength() : 0;

    return *avgEdgeLen_;
}

size_t ObjectMeshHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + data_.heapBytes()
        + MR::heapBytes( textures_ )
        + ancillaryTexture_.heapBytes()
        + ancillaryUVCoordinates_.heapBytes();
}

const char * ObjectMeshHolder::actualSerializeFormat() const
{
    return serializeFormat_ ? serializeFormat_ : defaultSerializeMeshFormat().c_str();
}

void ObjectMeshHolder::setSerializeFormat( const char * newFormat )
{
    if ( newFormat && *newFormat != '.' )
    {
        assert( false );
        return;
    }
    serializeFormat_ = newFormat;
}

void ObjectMeshHolder::resetFrontColor()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
}

void ObjectMeshHolder::resetColors()
{
    // cannot implement in the opposite way to keep `setDefaultColors_()` non-virtual
    setDefaultColors_();
}

size_t ObjectMeshHolder::numUndirectedEdges() const
{
    if ( !numUndirectedEdges_ )
        numUndirectedEdges_ = data_.mesh ? data_.mesh->topology.computeNotLoneUndirectedEdges() : 0;
    return *numUndirectedEdges_;
}

size_t ObjectMeshHolder::numHoles() const
{
    if ( !numHoles_ )
        numHoles_ = data_.mesh ? data_.mesh->topology.findNumHoles() : 0;
    return *numHoles_;
}

size_t ObjectMeshHolder::numComponents() const
{
    if ( !numComponents_ )
        numComponents_ = data_.mesh ? MeshComponents::getNumComponents( *data_.mesh ) : 0;
    return *numComponents_;
}

size_t ObjectMeshHolder::numHandles() const
{
    if ( !data_.mesh )
        return 0;
    int EulerCharacteristic = data_.mesh->topology.numValidFaces() + (int)numHoles() + data_.mesh->topology.numValidVerts() - (int)numUndirectedEdges();
    return numComponents() - EulerCharacteristic / 2;
}

void ObjectMeshHolder::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    VisualObject::setDirtyFlags( mask, invalidateCaches );

    if ( mask & DIRTY_FACE )
    {
        numHoles_.reset();
        numComponents_.reset();
        numUndirectedEdges_.reset();
        numHandles_.reset();
        meshIsClosed_.reset();
    }

    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
    {
        worldBox_.reset();
        worldBox_.get().reset();
        totalArea_.reset();
        selectedArea_.reset();
        volume_.reset();
        avgEdgeLen_.reset();
        if ( invalidateCaches && data_.mesh )
            data_.mesh->invalidateCaches();
    }
}

void ObjectMeshHolder::setCreases( UndirectedEdgeBitSet creases )
{
    if ( creases == data_.creases )
        return;
    data_.creases = std::move( creases );
    numCreaseEdges_.reset();
    creasesChangedSignal();
    if ( data_.creases.any() )
    {
        setDirtyFlags( DIRTY_CORNERS_RENDER_NORMAL );
    }
    else
    {
        setDirtyFlags( DIRTY_VERTS_RENDER_NORMAL );
    }
}

void ObjectMeshHolder::swapBase_( Object& other )
{
    if ( auto otherMesh = other.asType<ObjectMeshHolder>() )
        std::swap( *this, *otherMesh );
    else
        assert( false );
}

void ObjectMeshHolder::swapSignals_( Object& other )
{
    VisualObject::swapSignals_( other );
    if ( auto otherMesh = other.asType<ObjectMeshHolder>() )
    {
        std::swap( faceSelectionChangedSignal, otherMesh->faceSelectionChangedSignal );
        std::swap( edgeSelectionChangedSignal, otherMesh->edgeSelectionChangedSignal );
        std::swap( creasesChangedSignal, otherMesh->creasesChangedSignal );
    }
    else
        assert( false );
}

const ViewportProperty<Color>& ObjectMeshHolder::getSelectedEdgesColorsForAllViewports() const
{
    return edgeSelectionColor_;
}

void ObjectMeshHolder::setSelectedEdgesColorsForAllViewports( ViewportProperty<Color> val )
{
    edgeSelectionColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectMeshHolder::getSelectedFacesColorsForAllViewports() const
{
    return faceSelectionColor_;
}

void ObjectMeshHolder::setSelectedFacesColorsForAllViewports( ViewportProperty<Color> val )
{
    faceSelectionColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectMeshHolder::getBordersColorsForAllViewports() const
{
    return bordersColor_;
}

void ObjectMeshHolder::setBordersColorsForAllViewports( ViewportProperty<Color> val )
{
    bordersColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectMeshHolder::getEdgesColorsForAllViewports() const
{
    return edgesColor_;
}

void ObjectMeshHolder::setEdgesColorsForAllViewports( ViewportProperty<Color> val )
{
    edgesColor_ = std::move( val );
    needRedraw_ = true;
}

void ObjectMeshHolder::setDefaultSceneProperties_()
{
    setDefaultColors_();
    setFlatShading( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat );
}

// .PLY format is the most compact among other formats with zero compression costs
static std::string sDefaultSerializeMeshFormat = ".ply";

const std::string & defaultSerializeMeshFormat()
{
    return sDefaultSerializeMeshFormat;
}

void setDefaultSerializeMeshFormat( std::string newFormat )
{
    assert( !newFormat.empty() && newFormat[0] == '.' );
    sDefaultSerializeMeshFormat = std::move( newFormat );
}

} //namespace MR
