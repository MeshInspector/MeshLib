#include "MRMeshHolder.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRMeshSave.h"
#include "MRSerializer.h"
#include "MRMeshLoad.h"
#include "MRMeshNormals.h"
#include "MRSceneColors.h"
#include "MRMeshComponents.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MRIRenderObject.h"
#include "MRViewportId.h"
#include "MRGTest.h"
#include "MRSceneSettings.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( MeshHolder )

const Color& MeshHolder::getSelectedFacesColor() const
{
    return faceSelectionColor_;
}

const Color& MeshHolder::getSelectedEdgesColor() const
{
    return edgeSelectionColor_;
}

void MeshHolder::setSelectedFacesColor( const Color& color )
{
    if ( color == faceSelectionColor_ )
        return;
    faceSelectionColor_ = color;
}

void MeshHolder::setSelectedEdgesColor( const Color& color )
{
    if ( color == edgeSelectionColor_ )
        return;
    edgeSelectionColor_ = color;
}

tl::expected<std::future<void>, std::string> MeshHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !mesh_ )
        return {};

    return std::async( getAsyncLaunchType(),
        [mesh = mesh_, filename = path.u8string() + u8".ctm"]() { MR::MeshSave::toCtm( *mesh, filename ); } );
}

void MeshHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowFaces"] = showFaces_.value();
    root["ShowLines"] = showEdges_.value();
    root["ShowBordersHighlight"] = showBordersHighlight_.value();
    root["ShowSelectedEdges"] = showSelectedEdges_.value();
    root["FaceBased"] = !flatShading_.empty();
    // edges
    serializeToJson( Vector4f( edgesColor_ ), root["Colors"]["Edges"] );
    // borders
    serializeToJson( Vector4f( bordersColor_ ), root["Colors"]["Borders"] );

    serializeToJson( Vector4f( faceSelectionColor_ ), root["Colors"]["Selection"]["Diffuse"] );

    serializeToJson( selectedTriangles_, root["SelectionFaceBitSet"] );
    serializeToJson( selectedEdges_, root["SelectionEdgeBitSet"] );
    serializeToJson( creases_, root["MeshCreasesUndirEdgeBitSet"] );

    root["Type"].append( MeshHolder::TypeName() );
}

void MeshHolder::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    const auto& selectionColor = root["Colors"]["Selection"];

    if ( root["ShowFaces"].isUInt() )
        showFaces_ = ViewportMask{ root["ShowFaces"].asUInt() };
    if ( root["ShowLines"].isUInt() )
        showEdges_ = ViewportMask{ root["ShowLines"].asUInt() };
    if ( root["ShowBorderHighlight"].isUInt() )
        showBordersHighlight_ = ViewportMask{ root["ShowBorderHighlight"].asUInt() };
    if ( root["ShowSelectedEdges"].isUInt() )
        showSelectedEdges_ = ViewportMask{ root["ShowSelectedEdges"].asUInt() };
    if ( root["FaceBased"].isBool() ) // Support old versions
        flatShading_ = root["FaceBased"].asBool() ? ViewportMask::all() : ViewportMask{};

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

tl::expected<void, std::string> MeshHolder::deserializeModel_( const std::filesystem::path& path )
{
    auto res = MeshLoad::fromCtm( path.u8string() + u8".ctm" );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );

    mesh_ = std::make_shared<Mesh>( std::move( res.value() ) );
    return {};
}

Box3f MeshHolder::computeBoundingBox_() const
{
    if ( !mesh_ )
        return Box3f();
    return mesh_->computeBoundingBox();
}

Box3f MeshHolder::computeBoundingBoxXf_() const
{
    if ( !mesh_ )
        return Box3f();
    const auto tempXf = worldXf();
    return mesh_->computeBoundingBox( &tempXf );
}

Vector<MR::Vector3f, MR::VertId> MeshHolder::computeVertsNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerVertNormals( *mesh_ );
}

Vector<MR::Vector3f, MR::FaceId> MeshHolder::computeFacesNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerFaceNormals( *mesh_ );
}

Vector<MR::TriangleCornerNormals, MR::FaceId> MeshHolder::computeCornerNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerCornerNormals( *mesh_, creases_.any() ? &creases_ : nullptr );
}

const ViewportMask& MeshHolder::getVisualizePropertyMask( unsigned type ) const
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

void MeshHolder::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<MeshHolder>( *this );
}

void MeshHolder::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ) );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setSelectedFacesColor( SceneColors::get( SceneColors::SelectedFaces ) );
    setSelectedEdgesColor( SceneColors::get( SceneColors::SelectedEdges ) );
    setEdgesColor( SceneColors::get( SceneColors::Edges ) );
}

MeshHolder::MeshHolder( const MeshHolder& other ) :
    VisualObject( other )
{
    edgesColor_ = other.edgesColor_;
    bordersColor_ = other.bordersColor_;
    flatShading_ = other.flatShading_;
    showFaces_ = other.showFaces_;
    showEdges_ = other.showEdges_;
    showSelectedEdges_ = other.showSelectedEdges_;
    showBordersHighlight_ = other.showBordersHighlight_;
    facesColorMap_ = other.facesColorMap_;
    edgeWidth_ = other.edgeWidth_;

    selectedTriangles_ = other.selectedTriangles_;
    selectedEdges_ = other.selectedEdges_;
    faceSelectionColor_ = other.faceSelectionColor_;

    showSelectedFaces_ = other.showSelectedFaces_;
}

MeshHolder::MeshHolder()
{
    setDefaultColors_();
    setFlatShading( SceneSettings::get( SceneSettings::Type::MeshFlatShading ) );
}

uint32_t MeshHolder::getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const
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

bool MeshHolder::getRedrawFlag( ViewportMask viewportMask ) const
{
    return Object::getRedrawFlag( viewportMask ) ||
        ( isVisible( viewportMask ) &&
          ( dirty_ & ( ~( DIRTY_CACHES | ( DIRTY_RENDER_NORMALS - getNeededNormalsRenderDirtyValue( viewportMask ) ) ) ) ) );
}

void MeshHolder::resetDirtyExeptMask( uint32_t mask ) const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= ( DIRTY_CACHES | mask );
}

std::vector<std::string> MeshHolder::getInfoLines() const
{
    std::vector<std::string> res;

    res.push_back( "type : MeshHolder" );
    return res;
}

std::shared_ptr<Object> MeshHolder::clone() const
{
    auto res = std::make_shared<MeshHolder>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

std::shared_ptr<Object> MeshHolder::shallowClone() const
{
    auto res = std::make_shared<MeshHolder>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

void MeshHolder::selectFaces( FaceBitSet newSelection )
{
    selectedTriangles_ = std::move( newSelection );
    numSelectedFaces_.reset();
    dirty_ |= DIRTY_SELECTION;
}

void MeshHolder::selectEdges( const UndirectedEdgeBitSet& newSelection )
{
    selectedEdges_ = newSelection;
    numSelectedEdges_.reset();
    dirty_ |= DIRTY_EDGES_SELECTION;
}


bool MeshHolder::isMeshClosed() const
{
    if ( !meshIsClosed_ )
        meshIsClosed_ = mesh_ && mesh_->topology.isClosed();

    return *meshIsClosed_;
}

Box3f MeshHolder::getWorldBox() const
{
    if ( !mesh_ )
        return {};
    const auto worldXf = this->worldXf();
    if ( auto v = worldBox_.get( worldXf ) )
        return *v;
    const auto box = mesh_->computeBoundingBox( &worldXf );
    worldBox_.set( worldXf, box );
    return box;
}

size_t MeshHolder::numSelectedFaces() const
{
    if ( !numSelectedFaces_ )
        numSelectedFaces_ = selectedTriangles_.count();

    return *numSelectedFaces_;
}

size_t MeshHolder::numSelectedEdges() const
{
    if ( !numSelectedEdges_ )
        numSelectedEdges_ = selectedEdges_.count();

    return *numSelectedEdges_;
}

size_t MeshHolder::numCreaseEdges() const
{
    if ( !numCreaseEdges_ )
        numCreaseEdges_ = creases_.count();

    return *numCreaseEdges_;
}

double MeshHolder::totalArea() const
{
    if ( !totalArea_ )
        totalArea_ = mesh_ ? mesh_->area() : 0.0;

    return *totalArea_;
}

void MeshHolder::setDirtyFlags( uint32_t mask )
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
        totalArea_.reset();
        if ( mesh_ )
            mesh_->invalidateCaches();
    }
}

void MeshHolder::setCreases( UndirectedEdgeBitSet creases )
{
    if ( creases == creases_ )
        return;
    creases_ = std::move( creases );
    numCreaseEdges_.reset();

    if ( creases_.any() )
    {
        dirty_ |= DIRTY_CORNERS_NORMAL | DIRTY_CORNERS_RENDER_NORMAL;
    }
    else
    {
        dirty_ |= DIRTY_VERTS_NORMAL | DIRTY_VERTS_RENDER_NORMAL;
    }
}

void MeshHolder::swapBase_( Object& other )
{
    if ( auto otherMesh = other.asType<MeshHolder>() )
        std::swap( *this, *otherMesh );
    else
        assert( false );
}

AllVisualizeProperties MeshHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( MeshVisualizePropertyType::MeshVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const Vector<MR::Vector3f, MR::FaceId>& MeshHolder::getFacesNormals() const
{
    std::unique_lock lock( readCacheMutex_.getMutex() );
    if ( dirty_ & DIRTY_FACES_NORMAL )
    {
        facesNormalsCache_ = computeFacesNormals_();
        dirty_ &= ~DIRTY_FACES_NORMAL;
    }
    return facesNormalsCache_;
}

const Vector<MR::TriangleCornerNormals, MR::FaceId>& MeshHolder::getCornerNormals() const
{
    std::unique_lock lock( readCacheMutex_.getMutex() );
    if ( dirty_ & DIRTY_CORNERS_NORMAL )
    {
        cornerNormalsCache_ = computeCornerNormals_();
        dirty_ &= ~DIRTY_CORNERS_NORMAL;
    }
    return cornerNormalsCache_;
}

} //namespace MR
