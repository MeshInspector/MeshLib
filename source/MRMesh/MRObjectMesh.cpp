#include "MRObjectMesh.h"
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
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectMesh )

const Color& ObjectMesh::getSelectedFacesColor() const
{
    return faceSelectionColor_;
}

const Color& ObjectMesh::getSelectedEdgesColor() const
{
    return edgeSelectionColor_;
}

void ObjectMesh::setSelectedFacesColor( const Color& color )
{
    if( color == faceSelectionColor_ )
        return;
    faceSelectionColor_ = color;
}

void ObjectMesh::setSelectedEdgesColor( const Color& color )
{
    if( color == edgeSelectionColor_ )
        return;
    edgeSelectionColor_ = color;
}

void ObjectMesh::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ) );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setSelectedFacesColor( SceneColors::get( SceneColors::SelectedFaces ) );
    setSelectedEdgesColor( SceneColors::get( SceneColors::SelectedEdges ) );
    setEdgesColor( SceneColors::get( SceneColors::Edges ) );
}

tl::expected<std::future<void>, std::string> ObjectMesh::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !mesh_ )
        return {};

    return std::async( std::launch::async, 
        [mesh = mesh_, filename = path.u8string() + u8".ctm"]() { MR::MeshSave::toCtm( *mesh, filename ); } );
}

void ObjectMesh::serializeFields_( Json::Value& root ) const
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

    root["Type"].append( ObjectMesh::TypeName() );
}

void ObjectMesh::deserializeFields_( const Json::Value& root )
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

tl::expected<void, std::string> ObjectMesh::deserializeModel_( const std::filesystem::path& path )
{
    auto res = MeshLoad::fromCtm( path.u8string() + u8".ctm" );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );
    
    mesh_ = std::make_shared<Mesh>(std::move( res.value() ));
    return {};
}

Box3f ObjectMesh::computeBoundingBox_() const
{
    if ( !mesh_ )
        return Box3f();
    return mesh_->computeBoundingBox();
}

Box3f ObjectMesh::computeBoundingBoxXf_() const
{
    if( !mesh_ )
        return Box3f();
    const auto tempXf = worldXf();
    return mesh_->computeBoundingBox( &tempXf );
}

Vector<MR::Vector3f, MR::VertId> ObjectMesh::computeVertsNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerVertNormals( *mesh_ );
}

Vector<MR::Vector3f, MR::FaceId> ObjectMesh::computeFacesNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerFaceNormals( *mesh_ );
}

Vector<MR::TriangleCornerNormals, MR::FaceId> ObjectMesh::computeCornerNormals_() const
{
    if ( !mesh_ )
        return {};
    return computePerCornerNormals( *mesh_, creases_.any() ? &creases_ : nullptr );
}

const ViewportMask& ObjectMesh::getVisualizePropertyMask( unsigned type ) const
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
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectMesh::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectMesh>( *this );
}

ObjectMesh::ObjectMesh()
{
    setDefaultColors_();
}

ObjectMesh::ObjectMesh( const ObjectMesh& other ) :
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

uint32_t ObjectMesh::getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const
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

bool ObjectMesh::getRedrawFlag( ViewportMask viewportMask ) const
{
    return Object::getRedrawFlag( viewportMask ) ||
        ( isVisible( viewportMask ) &&
          ( dirty_ & ( ~( DIRTY_CACHES | ( DIRTY_RENDER_NORMALS - getNeededNormalsRenderDirtyValue( viewportMask ) ) ) ) ) );
}

void ObjectMesh::resetDirtyExeptMask( uint32_t mask ) const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= ( DIRTY_CACHES | mask );
}

std::optional<MeshIntersectionResult> ObjectMesh::worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region ) const
{
    std::optional<MeshIntersectionResult> res;
    if ( !mesh_ )
        return res;
    const AffineXf3f rayToMeshXf = worldXf().inverse();
    res = rayMeshIntersect( { *mesh_, region }, transformed( worldRay, rayToMeshXf ) );
    return res;
}

bool ObjectMesh::isMeshClosed() const
{
    if ( !meshIsClosed_ )
        meshIsClosed_ = mesh_ && mesh_->topology.isClosed();

    return *meshIsClosed_;
}

size_t ObjectMesh::numSelectedFaces() const
{
    if ( !numSelectedFaces_ )
        numSelectedFaces_ = selectedTriangles_.count();

    return *numSelectedFaces_;
}

size_t ObjectMesh::numSelectedEdges() const
{
    if ( !numSelectedEdges_ )
        numSelectedEdges_ = selectedEdges_.count();

    return *numSelectedEdges_;
}

void ObjectMesh::applyScale( float scaleFactor )
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

void ObjectMesh::setMesh( std::shared_ptr< Mesh > mesh )
{
    mesh_ = std::move(mesh);
    selectFaces({});
    selectEdges({});
    setDirtyFlags( DIRTY_ALL );
}

void ObjectMesh::selectFaces( FaceBitSet newSelection )
{
    selectedTriangles_ = std::move( newSelection );
    numSelectedFaces_.reset();
    dirty_ |= DIRTY_SELECTION;
}

void ObjectMesh::selectEdges(const UndirectedEdgeBitSet& newSelection)
{
    selectedEdges_ = newSelection;
    numSelectedEdges_.reset();
    dirty_ |= DIRTY_EDGES_SELECTION;
}

std::vector<std::string> ObjectMesh::getInfoLines() const
{
    std::vector<std::string> res;

    res.push_back( "type : ObjectMesh" );
    if ( mesh_ )
    {
        if ( !meshStat_ )
        {
            MeshStat ms;
            ms.numComponents = MeshComponents::getNumComponents( *mesh_ );
            ms.numUndirectedEdges = mesh_->topology.computeNotLoneUndirectedEdges();
            ms.numHoles = mesh_->topology.findHoleRepresentiveEdges().size();
            meshStat_ = ms;
        }

        res.push_back( "components: " + std::to_string( meshStat_->numComponents ) );
        res.push_back( "vertices: " + std::to_string( mesh_->topology.numValidVerts() ) );

        res.push_back( "faces: " + std::to_string( mesh_->topology.numValidFaces() ) );
        if( auto nFacesSelected = numSelectedFaces() )
            res.back() += " / " + std::to_string( nFacesSelected ) + " selected";

        res.push_back( "edges: " + std::to_string( meshStat_->numUndirectedEdges ) );
        if( auto nEdgesSelected = numSelectedEdges() )
            res.back() += " / " + std::to_string( nEdgesSelected ) + " selected";

        res.push_back( "holes: " + std::to_string( meshStat_->numHoles ) );

        boundingBoxToInfoLines_( res );
    }
    else
        res.push_back( "no mesh" );
    return res;
}

void ObjectMesh::showSelectedFaces( bool show )
{
    if ( show == showSelectedFaces_ )
        return;
    showSelectedFaces_ = show;
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

void ObjectMesh::setDirtyFlags( uint32_t mask )
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

    if ( ( mask & DIRTY_POSITION || mask & DIRTY_FACE ) && mesh_ )
    {
        mesh_->invalidateCaches();
        meshChangedSignal( mask );
    }
}

void ObjectMesh::setCreases( UndirectedEdgeBitSet creases )
{
    if ( creases == creases_ )
        return;
    creases_ = std::move( creases );
    dirty_ |= ( DIRTY_CORNERS_NORMAL | DIRTY_CORNERS_RENDER_NORMAL );
}

void ObjectMesh::swap( Object& other )
{
    if ( auto otherMesh = other.asType<ObjectMesh>() )
    {
        std::swap( *this, *otherMesh );
        // swap signals second time to return in place
        std::swap( meshChangedSignal, otherMesh->meshChangedSignal );
    }
    else
        assert( false );
}

AllVisualizeProperties ObjectMesh::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( MeshVisualizePropertyType::MeshVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const Vector<MR::Vector3f, MR::FaceId>& ObjectMesh::getFacesNormals() const
{
    std::unique_lock lock( readCacheMutex_.getMutex() );
    if ( dirty_ & DIRTY_FACES_NORMAL )
    {
        facesNormalsCache_ = computeFacesNormals_();
        dirty_ &= ~DIRTY_FACES_NORMAL;
    }
    return facesNormalsCache_;
}

const Vector<MR::TriangleCornerNormals, MR::FaceId>& ObjectMesh::getCornerNormals() const
{
    std::unique_lock lock( readCacheMutex_.getMutex() );
    if ( dirty_ & DIRTY_CORNERS_NORMAL )
    {
        cornerNormalsCache_ = computeCornerNormals_();
        dirty_ &= ~DIRTY_CORNERS_NORMAL;
    }
    return cornerNormalsCache_;
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
