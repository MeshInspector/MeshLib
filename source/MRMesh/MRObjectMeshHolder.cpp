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
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRStringConvert.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMeshHolder )

const Color& ObjectMeshHolder::getSelectedFacesColor() const
{
    return faceSelectionColor_;
}

const Color& ObjectMeshHolder::getSelectedEdgesColor() const
{
    return edgeSelectionColor_;
}

void ObjectMeshHolder::setSelectedFacesColor( const Color& color )
{
    if ( color == faceSelectionColor_ )
        return;
    faceSelectionColor_ = color;
}

void ObjectMeshHolder::setSelectedEdgesColor( const Color& color )
{
    if ( color == edgeSelectionColor_ )
        return;
    edgeSelectionColor_ = color;
}

tl::expected<std::future<void>, std::string> ObjectMeshHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !mesh_ )
        return {};

#ifndef MRMESH_NO_OPENCTM
    auto save = [mesh = mesh_, filename = utf8string( path ) + ".ctm", this]()
    { 
        MR::MeshSave::toCtm( *mesh, filename, {}, vertsColorMap_.empty() ? nullptr : &vertsColorMap_ );
    };
#else
    auto save = [mesh = mesh_, filename = utf8string( path ) + ".mrmesh", this]()
    {
        MR::MeshSave::toMrmesh( *mesh, filename );
    };
#endif

    return std::async( getAsyncLaunchType(), save );
}

void ObjectMeshHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowFaces"] = showFaces_.value();
    root["ShowLines"] = showEdges_.value();
    root["ShowBordersHighlight"] = showBordersHighlight_.value();
    root["ShowSelectedEdges"] = showSelectedEdges_.value();
    root["ShowSelectedFaces"] = showSelectedFaces_.value();
    root["OnlyOddFragments"] = onlyOddFragments_.value();
    root["FaceBased"] = !flatShading_.empty();
    root["ColoringType"] = ( coloringType_ == ColoringType::VertsColorMap ) ? "PerVertex" : "Solid";

    // edges
    serializeToJson( Vector4f( edgesColor_ ), root["Colors"]["Edges"] );
    // borders
    serializeToJson( Vector4f( bordersColor_ ), root["Colors"]["Borders"] );

    serializeToJson( Vector4f( faceSelectionColor_ ), root["Colors"]["Selection"]["Diffuse"] );

    serializeToJson( selectedTriangles_, root["SelectionFaceBitSet"] );
    serializeToJson( selectedEdges_, root["SelectionEdgeBitSet"] );
    serializeToJson( creases_, root["MeshCreasesUndirEdgeBitSet"] );

    root["Type"].append( ObjectMeshHolder::TypeName() );
}

void ObjectMeshHolder::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    const auto& selectionColor = root["Colors"]["Selection"];

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
    if ( root["FaceBased"].isBool() ) // Support old versions
        flatShading_ = root["FaceBased"].asBool() ? ViewportMask::all() : ViewportMask{};
    if ( root["ColoringType"].isString() )
    {
        const auto stype = root["ColoringType"].asString();
        if ( stype == "PerVertex" )
            setColoringType( ColoringType::VertsColorMap );
    }

    Vector4f resVec;
    deserializeFromJson( selectionColor["Diffuse"], resVec );
    faceSelectionColor_ = Color( resVec );
    // edges
    deserializeFromJson( root["Colors"]["Edges"], resVec );
    edgesColor_ = Color( resVec );
    // borders
    deserializeFromJson( root["Colors"]["Borders"], resVec );
    bordersColor_ = Color( resVec );

    deserializeFromJson( root["SelectionFaceBitSet"], selectedTriangles_ );
    deserializeFromJson( root["SelectionEdgeBitSet"], selectedEdges_ );
    deserializeFromJson( root["MeshCreasesUndirEdgeBitSet"], creases_ );
}

tl::expected<void, std::string> ObjectMeshHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    vertsColorMap_.clear();
#ifndef MRMESH_NO_OPENCTM
    auto res = MeshLoad::fromCtm( utf8string( path ) + ".ctm", &vertsColorMap_, progressCb );
#else
    auto res = MeshLoad::fromMrmesh( utf8string( path ) + ".mrmesh", &vertsColorMap_, progressCb );
#endif
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );

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
    switch ( MeshVisualizePropertyType::Type( type ) )
    {
    case MR::MeshVisualizePropertyType::Faces:
        return showFaces_;
    case MR::MeshVisualizePropertyType::Edges:
        return showEdges_;
    case MR::MeshVisualizePropertyType::FlatShading:
        return flatShading_;
    case MR::MeshVisualizePropertyType::OnlyOddFragments:
        return onlyOddFragments_;
    case MR::MeshVisualizePropertyType::BordersHighlight:
        return showBordersHighlight_;
    case MR::MeshVisualizePropertyType::SelectedEdges:
        return showSelectedEdges_;
    case MR::MeshVisualizePropertyType::SelectedFaces:
        return showSelectedFaces_;
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectMeshHolder::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectMeshHolder>( *this );
}

void ObjectMeshHolder::updateMeshStat_() const
{
    if ( !meshStat_ )
    {
        MeshStat ms;
        ms.numComponents = MeshComponents::getNumComponents( *mesh_ );
        ms.numUndirectedEdges = mesh_->topology.computeNotLoneUndirectedEdges();
        ms.numHoles = mesh_->topology.findHoleRepresentiveEdges().size();
        meshStat_ = ms;
    }
}

void ObjectMeshHolder::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setSelectedFacesColor( SceneColors::get( SceneColors::SelectedFaces ) );
    setSelectedEdgesColor( SceneColors::get( SceneColors::SelectedEdges ) );
    setEdgesColor( SceneColors::get( SceneColors::Edges ) );
}

ObjectMeshHolder::ObjectMeshHolder( const ObjectMeshHolder& other ) :
    VisualObject( other )
{
    selectedTriangles_ = other.selectedTriangles_;
    selectedEdges_ = other.selectedEdges_;
    creases_ = other.creases_;

    showFaces_ = other.showFaces_;
    showEdges_ = other.showEdges_;
    showSelectedEdges_ = other.showSelectedEdges_;
    showSelectedFaces_ = other.showSelectedFaces_;
    showBordersHighlight_ = other.showBordersHighlight_;
    flatShading_ = other.flatShading_;
    onlyOddFragments_ = other.onlyOddFragments_;
    
    edgesColor_ = other.edgesColor_;
    bordersColor_ = other.bordersColor_;
    edgeSelectionColor_ = other.edgeSelectionColor_;
    faceSelectionColor_ = other.faceSelectionColor_;

    facesColorMap_ = other.facesColorMap_;
    edgeWidth_ = other.edgeWidth_;
}

ObjectMeshHolder::ObjectMeshHolder()
{
    setDefaultColors_();
    setFlatShading( SceneSettings::get( SceneSettings::Type::MeshFlatShading ) );
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
    return mesh_ && mesh_->topology.numValidFaces() != 0;
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
    dirty_ |= DIRTY_SELECTION;
}

void ObjectMeshHolder::selectEdges( UndirectedEdgeBitSet newSelection )
{
    selectedEdges_ = std::move( newSelection );
    numSelectedEdges_.reset();
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
        numSelectedFaces_ = selectedTriangles_.count();

    return *numSelectedFaces_;
}

size_t ObjectMeshHolder::numSelectedEdges() const
{
    if ( !numSelectedEdges_ )
        numSelectedEdges_ = selectedEdges_.count();

    return *numSelectedEdges_;
}

size_t ObjectMeshHolder::numCreaseEdges() const
{
    if ( !numCreaseEdges_ )
        numCreaseEdges_ = creases_.count();

    return *numCreaseEdges_;
}

double ObjectMeshHolder::totalArea() const
{
    if ( !totalArea_ )
        totalArea_ = mesh_ ? mesh_->area() : 0.0;

    return *totalArea_;
}

size_t ObjectMeshHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + selectedTriangles_.heapBytes()
        + selectedEdges_.heapBytes()
        + creases_.heapBytes()
        + facesColorMap_.heapBytes()
        + MR::heapBytes( mesh_ );
}

size_t ObjectMeshHolder::numHoles() const
{
    updateMeshStat_();
    return meshStat_->numHoles;
}

void ObjectMeshHolder::setDirtyFlags( uint32_t mask )
{
    // selected faces and edges can be changed only by the methods of this class, 
    // which set dirty flags appropriately
    mask &= ~( DIRTY_SELECTION | DIRTY_EDGES_SELECTION );

    VisualObject::setDirtyFlags( mask );

    if ( mask & DIRTY_FACE )
    {
        meshStat_.reset();
        meshIsClosed_.reset();
    }

    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
    {
        worldBox_.reset();
        worldBox_.get().reset();
        totalArea_.reset();
        if ( mesh_ )
            mesh_->invalidateCaches();
    }
}

void ObjectMeshHolder::setCreases( UndirectedEdgeBitSet creases )
{
    if ( creases == creases_ )
        return;
    creases_ = std::move( creases );
    numCreaseEdges_.reset();

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

AllVisualizeProperties ObjectMeshHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( MeshVisualizePropertyType::MeshVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

} //namespace MR
