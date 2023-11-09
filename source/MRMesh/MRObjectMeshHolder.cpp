#include "MRObjectMeshHolder.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRMeshSave.h"
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
#include "MRPch/MRJson.h"
#include "MRPch/MRAsyncLaunchType.h"
#include <filesystem>

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

Expected<std::future<void>, std::string> ObjectMeshHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !mesh_ )
        return {};

#ifndef MRMESH_NO_OPENCTM
    auto save = [mesh = mesh_, filename = utf8string( path ) + ".ctm", this]()
    { 
        MR::MeshSave::toCtm( *mesh, pathFromUtf8( filename ), {}, vertsColorMap_.empty() ? nullptr : &vertsColorMap_ );
    };
#else
    auto save = [mesh = mesh_, filename = utf8string( path ) + ".mrmesh"]()
    {
        MR::MeshSave::toMrmesh( *mesh, pathFromUtf8( filename ) );
    };
#endif

    return std::async( getAsyncLaunchType(), save );
}

void ObjectMeshHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowTexture"] = showTexture_.value();
    root["ShowFaces"] = showFaces_.value();
    root["ShowLines"] = showEdges_.value();
    root["ShowBordersHighlight"] = showBordersHighlight_.value();
    root["ShowSelectedEdges"] = showSelectedEdges_.value();
    root["ShowSelectedFaces"] = showSelectedFaces_.value();
    root["OnlyOddFragments"] = onlyOddFragments_.value();
    root["PolygonOffset"] = polygonOffset_.value();
    root["ShadingEnabled"] = shadingEnabled_.value();
    root["FaceBased"] = !flatShading_.empty();
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
    serializeToJson( facesColorMap_.vec_, root["FaceColors"] );

    // texture
    serializeToJson( texture_, root["Texture"] );
    serializeToJson( uvCoordinates_.vec_, root["UVCoordinates"] );
    // edges
    serializeToJson( Vector4f( edgesColor_.get() ), root["Colors"]["Edges"] );
    // borders
    serializeToJson( Vector4f( bordersColor_.get() ), root["Colors"]["Borders"] );

    serializeToJson( Vector4f( faceSelectionColor_.get() ), root["Colors"]["Selection"]["Diffuse"] );

    serializeToJson( selectedTriangles_, root["SelectionFaceBitSet"] );
    if ( mesh_ )
    {
        serializeViaVerticesToJson( selectedEdges_, mesh_->topology, root["SelectionEdgeBitSet"] );
        serializeViaVerticesToJson( creases_, mesh_->topology, root["MeshCreasesUndirEdgeBitSet"] );
    }
    else
    {
        serializeToJson( selectedEdges_, root["SelectionEdgeBitSet"] );
        serializeToJson( creases_, root["MeshCreasesUndirEdgeBitSet"] );
    }

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
    deserializeFromJson( root["FaceColors"], facesColorMap_.vec_ );

    Vector4f resVec;
    deserializeFromJson( selectionColor["Diffuse"], resVec );
    faceSelectionColor_.set( Color( resVec ) );
    // texture
    if ( root["Texture"].isObject() )
        deserializeFromJson( root["Texture"], texture_ );
    if ( root["UVCoordinates"].isObject() )
        deserializeFromJson( root["UVCoordinates"], uvCoordinates_.vec_ );
    // edges
    deserializeFromJson( root["Colors"]["Edges"], resVec );
    edgesColor_.set( Color( resVec ) );
    // borders
    deserializeFromJson( root["Colors"]["Borders"], resVec );
    bordersColor_.set( Color( resVec ) );

    deserializeFromJson( root["SelectionFaceBitSet"], selectedTriangles_ );

    if ( mesh_ )
    {
        selectedTriangles_ &= mesh_->topology.getValidFaces();

        const auto notLoneEdges = mesh_->topology.findNotLoneUndirectedEdges();
        deserializeViaVerticesFromJson( root["SelectionEdgeBitSet"], selectedEdges_, mesh_->topology );
        selectedEdges_ &= notLoneEdges;

        deserializeViaVerticesFromJson( root["MeshCreasesUndirEdgeBitSet"], creases_, mesh_->topology );
        creases_ &= notLoneEdges;
    }
    else
    {
        deserializeFromJson( root["SelectionEdgeBitSet"], selectedEdges_ );
        deserializeFromJson( root["MeshCreasesUndirEdgeBitSet"], creases_ );
    }
}

VoidOrErrStr ObjectMeshHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    vertsColorMap_.clear();
#ifndef MRMESH_NO_OPENCTM
    auto res = MeshLoad::fromCtm( pathFromUtf8( utf8string( path ) + ".ctm" ), { .colors = &vertsColorMap_, .callback = progressCb } );
#else
    auto res = MeshLoad::fromMrmesh( pathFromUtf8( utf8string( path ) + ".mrmesh" ), &vertsColorMap_, progressCb );
#endif
    if ( !res.has_value() )
        return unexpected( res.error() );

    mesh_ = std::make_shared<Mesh>( std::move( res.value() ) );
    return {};
}

Box3f ObjectMeshHolder::computeBoundingBox_() const
{
    if ( !mesh_ )
        return Box3f();
    return mesh_->computeBoundingBox();
}

const ViewportMask& ObjectMeshHolder::getVisualizePropertyMask( unsigned type ) const
{
    switch ( type )
    {
    case MeshVisualizePropertyType::Faces:
        return showFaces_;
    case MeshVisualizePropertyType::Texture:
        return showTexture_;
    case MeshVisualizePropertyType::Edges:
        return showEdges_;
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
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
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
}

ObjectMeshHolder::ObjectMeshHolder()
{
    setDefaultColors_();
    setFlatShading( SceneSettings::get( SceneSettings::Type::MeshFlatShading ) );
}

void ObjectMeshHolder::copyTextureAndColors( const ObjectMeshHolder & src, const VertMap & thisToSrc )
{
    MR_TIMER
    copyColors( src, thisToSrc );
    setTexture( src.getTexture() );

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
        res |= ( dirty_ & DIRTY_FACES_RENDER_NORMAL );
    }
    if ( ( flatShading & viewportMask ) != viewportMask )
    {
        if ( !creases_.any() )
        {
            res |= ( dirty_ & DIRTY_VERTS_RENDER_NORMAL );
        }
        else
        {
            res |= ( dirty_ & DIRTY_CORNERS_RENDER_NORMAL );
        }
    }
    return res;
}

bool ObjectMeshHolder::getRedrawFlag( ViewportMask viewportMask ) const
{
    return Object::getRedrawFlag( viewportMask ) ||
        ( isVisible( viewportMask ) &&
          ( dirty_ & ( ~( DIRTY_CACHES | ( DIRTY_RENDER_NORMALS - getNeededNormalsRenderDirtyValue( viewportMask ) ) ) ) ) );
}

void ObjectMeshHolder::resetDirtyExeptMask( uint32_t mask ) const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= ( DIRTY_CACHES | mask );
}


void ObjectMeshHolder::applyScale( float scaleFactor )
{
    if ( !mesh_ )
        return;

    auto& points = mesh_->points;

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
    return mesh_ && numUndirectedEdges() > 0;
}

std::shared_ptr<Object> ObjectMeshHolder::clone() const
{
    auto res = std::make_shared<ObjectMeshHolder>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

std::shared_ptr<Object> ObjectMeshHolder::shallowClone() const
{
    auto res = std::make_shared<ObjectMeshHolder>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

void ObjectMeshHolder::selectFaces( FaceBitSet newSelection )
{
    selectedTriangles_ = std::move( newSelection );
    numSelectedFaces_.reset();
    selectedArea_.reset();
    faceSelectionChangedSignal();
    dirty_ |= DIRTY_SELECTION;
}

void ObjectMeshHolder::selectEdges( UndirectedEdgeBitSet newSelection )
{
    selectedEdges_ = std::move( newSelection );
    numSelectedEdges_.reset();
    edgeSelectionChangedSignal();
    dirty_ |= DIRTY_EDGES_SELECTION;
}

bool ObjectMeshHolder::isMeshClosed() const
{
    if ( !meshIsClosed_ )
        meshIsClosed_ = mesh_ && mesh_->topology.isClosed();

    return *meshIsClosed_;
}

Box3f ObjectMeshHolder::getWorldBox( ViewportId id ) const
{
    if ( !mesh_ )
        return {};
    bool isDef = true;
    const auto worldXf = this->worldXf( id, &isDef );
    if ( isDef )
        id = {};
    auto & cache = worldBox_[id];
    if ( auto v = cache.get( worldXf ) )
        return *v;
    const auto box = mesh_->computeBoundingBox( &worldXf );
    cache.set( worldXf, box );
    return box;
}

size_t ObjectMeshHolder::numSelectedFaces() const
{
    if ( !numSelectedFaces_ )
    {
        numSelectedFaces_ = selectedTriangles_.count();
#ifndef NDEBUG
        // check that there are no selected invalid faces
        assert( !mesh_ || !( selectedTriangles_ - mesh_->topology.getValidFaces() ).any() );
#endif
    }

    return *numSelectedFaces_;
}

size_t ObjectMeshHolder::numSelectedEdges() const
{
    if ( !numSelectedEdges_ )
    {
        numSelectedEdges_ = selectedEdges_.count();
#ifndef NDEBUG
        // check that there are no selected invalid edges
        assert( !mesh_ || !( selectedEdges_ - mesh_->topology.findNotLoneUndirectedEdges() ).any() );
#endif
    }

    return *numSelectedEdges_;
}

size_t ObjectMeshHolder::numCreaseEdges() const
{
    if ( !numCreaseEdges_ )
    {
        numCreaseEdges_ = creases_.count();
#ifndef NDEBUG
        // check that there are no invalid edges among creases
        assert( !mesh_ || !( creases_ - mesh_->topology.findNotLoneUndirectedEdges() ).any() );
#endif
    }

    return *numCreaseEdges_;
}

double ObjectMeshHolder::totalArea() const
{
    if ( !totalArea_ )
        totalArea_ = mesh_ ? mesh_->area() : 0.0;

    return *totalArea_;
}

double ObjectMeshHolder::selectedArea() const
{
    if ( !selectedArea_ )
        selectedArea_ = mesh_ ? mesh_->area( &selectedTriangles_ ) : 0.0;

    return *selectedArea_;
}

double ObjectMeshHolder::volume() const
{
    if ( !volume_ )
        volume_ = mesh_ ? mesh_->volume() : 0.0;

    return *volume_;
}

float ObjectMeshHolder::avgEdgeLen() const
{
    if ( !avgEdgeLen_ )
        avgEdgeLen_ = mesh_ ? mesh_->averageEdgeLength() : 0;

    return *avgEdgeLen_;
}

size_t ObjectMeshHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + selectedTriangles_.heapBytes()
        + selectedEdges_.heapBytes()
        + creases_.heapBytes()
        + texture_.heapBytes()
        + ancillaryTexture_.heapBytes()
        + uvCoordinates_.heapBytes()
        + ancillaryUVCoordinates_.heapBytes()
        + facesColorMap_.heapBytes()
        + MR::heapBytes( mesh_ );
}

size_t ObjectMeshHolder::numUndirectedEdges() const
{
    if ( !numUndirectedEdges_ )
        numUndirectedEdges_ = mesh_ ? mesh_->topology.computeNotLoneUndirectedEdges() : 0;
    return *numUndirectedEdges_;
}

size_t ObjectMeshHolder::numHoles() const
{
    if ( !numHoles_ )
        numHoles_ = mesh_ ? mesh_->topology.findNumHoles() : 0;
    return *numHoles_;
}

size_t ObjectMeshHolder::numComponents() const
{
    if ( !numComponents_ )
        numComponents_ = mesh_ ? MeshComponents::getNumComponents( *mesh_ ) : 0;
    return *numComponents_;
}

size_t ObjectMeshHolder::numHandles() const
{
    if ( !mesh_ )
        return 0;
    int EulerCharacteristic = mesh_->topology.numValidFaces() + (int)numHoles() + mesh_->topology.numValidVerts() - (int)numUndirectedEdges();
    return numComponents() - EulerCharacteristic / 2;
}

void ObjectMeshHolder::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    // selected faces and edges can be changed only by the methods of this class, 
    // which set dirty flags appropriately
    mask &= ~( DIRTY_SELECTION | DIRTY_EDGES_SELECTION );

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
        if ( invalidateCaches && mesh_ )
            mesh_->invalidateCaches();
    }
}

void ObjectMeshHolder::setCreases( UndirectedEdgeBitSet creases )
{
    if ( creases == creases_ )
        return;
    creases_ = std::move( creases );
    numCreaseEdges_.reset();
    creasesChangedSignal();
    if ( creases_.any() )
    {
        dirty_ |= DIRTY_CORNERS_RENDER_NORMAL;
    }
    else
    {
        dirty_ |= DIRTY_VERTS_RENDER_NORMAL;
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

AllVisualizeProperties ObjectMeshHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( MeshVisualizePropertyType::MeshVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
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

} //namespace MR
