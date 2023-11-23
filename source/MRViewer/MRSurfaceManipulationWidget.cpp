#include "MRSurfaceManipulationWidget.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRViewerInstance.h"
#include "MRAppendHistory.h"
#include "MRViewer/MRGladGlfw.h"

#include "MRMouse.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRSurfaceDistance.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MREnumNeighbours.h"
#include "MRMesh/MRTimer.h"
#include <MRMesh/MRMeshRelax.h>
#include <MRMesh/MRBitSetParallelFor.h>


namespace MR
{
//const float k = r < 1-a ? std::sqrt( sqr( 1 - a ) - sqr( r ) ) + ( 1 - a ) : -std::sqrt( sqr( a ) - sqr( r - 1 ) ) + a; // alternative version F_point_shift(r,i) (i == a)

void SurfaceManipulationWidget::init( const std::shared_ptr<ObjectMesh>& objectMesh )
{
    obj_ = objectMesh;
    diagonal_ = obj_->getBoundingBox().diagonal();
    minRadius_ = obj_->avgEdgeLen() * 2.f;

    if ( firstInit_ )
    {
        settings_.radius = diagonal_ * 0.02f;
        settings_.editForce = diagonal_ * 0.01f;
        settings_.relaxForce = 0.2f;
        settings_.workMode = WorkMode::Add;
        firstInit_ = false;
    }
    settings_.radius = std::max( settings_.radius, minRadius_ );
    settings_.editForce = std::max( settings_.editForce, 0.001f );


    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    singleEditingRegion_ = VertBitSet( numV, false );
    visualizationRegion_ = VertBitSet( numV, false );
    generalEditingRegion_ = VertBitSet( numV, false );
    pointsShift_ = VertScalars( numV, 0.f );
    editingDistanceMap_ = VertScalars( numV, 0.f );
    visualizationDistanceMap_ = VertScalars( numV, 0.f );




    obj_->setAncillaryTexture( { { { Color { 255, 64, 64, 255 }, Color { 0, 0, 0, 0 } }, Vector2i { 1, 2 } } } );
    uvs_ = VertUVCoords( numV, { 0, 1 } );
    obj_->setAncillaryUVCoords( uvs_ );

    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceManipulationWidget::reset()
{
    if ( oldMesh_ )
        oldMesh_->detachFromParent();
    oldMesh_.reset();

    obj_->clearAncillaryTexture();
    obj_.reset();

    singleEditingRegion_.clear();
    visualizationRegion_.clear();
    generalEditingRegion_.clear();
    pointsShift_.clear();
    editingDistanceMap_.clear();
    visualizationDistanceMap_.clear();

    uvs_ = {};

    changeMeshAction_.reset();

    disconnect();
}

void SurfaceManipulationWidget::setSettings( const Settings& settings )
{
    settings_ = settings;
    settings_.radius = std::max( settings_.radius, minRadius_ );
    settings_.editForce = std::max( settings_.editForce, 0.001f );
    settings_.relaxForce = std::clamp( settings_.relaxForce, 0.001f, 0.5f );
    settings_.sharpness = std::clamp( settings_.sharpness, 1.f, 100.f );
    updateRegion_( mousePos_ );
}

bool SurfaceManipulationWidget::onMouseDown_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left )
        return false;

    auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
    if ( !obj || obj != obj_ )
        return false;

    oldMesh_ = std::dynamic_pointer_cast<ObjectMesh>( obj_->clone() );
    oldMesh_->setAncillary( true );
    oldMesh_->setGlobalAlpha( 1 );
    obj_->setPickable( false );
    obj_->parent()->addChild( oldMesh_ );

    changeMeshAction_ = std::make_shared<ChangeMeshAction>( "Change mesh surface", obj_ );
    mousePressed_ = true;
    timePoint_ = std::chrono::high_resolution_clock::now();
    changeSurface_();

    return true;
}

bool SurfaceManipulationWidget::onMouseUp_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left )
        return false;

    mousePressed_ = false;
    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    pointsShift_ = VertScalars( numV, 0.f );

    if ( settings_.workMode != WorkMode::Relax && settings_.relaxAfterEdit )
    {
        MeshRelaxParams params;
        params.region = &generalEditingRegion_;
        params.force = settings_.relaxForce;
        params.iterations = 5;
        relax( *obj_->varMesh(), params );
        obj_->setDirtyFlags( DIRTY_POSITION );
    }
    generalEditingRegion_ = VertBitSet( numV, false );

    if ( oldMesh_ )
        oldMesh_->detachFromParent();
    oldMesh_.reset();
    obj_->setPickable( true );
    AppendHistory( changeMeshAction_ );
    changeMeshAction_.reset();

    return true;
}

bool SurfaceManipulationWidget::onMouseMove_( int mouse_x, int mouse_y )
{
    updateRegion_( Vector2f{ float( mouse_x ), float( mouse_y ) } );
    if ( mousePressed_ )
        changeSurface_();

    return true;
}

void SurfaceManipulationWidget::changeSurface_()
{
    if ( !singleEditingRegion_.any() )
        return;

    MR_TIMER;

    if ( settings_.workMode == WorkMode::Relax )
    {
        MeshRelaxParams params;
        params.region = &singleEditingRegion_;
        params.force = settings_.relaxForce ;
        relax( *obj_->varMesh(), params );
        obj_->setDirtyFlags( DIRTY_POSITION );
        return;
    }

    Vector3f normal;
    const auto& mesh = *obj_->mesh();
    for ( auto v : singleEditingRegion_ )
        normal += mesh.normal( v );
    normal = normal.normalized();

    auto& points = obj_->varMesh()->points;

    const float maxShift = settings_.editForce;
    const float intensity = ( 101.f - settings_.sharpness ) / 100.f * 0.15f + 0.4f;
    const float a1 = -1.f * ( 1 - intensity ) / intensity / intensity;
    const float a2 = intensity / ( 1 - intensity ) / ( 1 - intensity );
    const float direction = settings_.workMode == WorkMode::Remove ? -1.f : 1.f;
    BitSetParallelFor( singleEditingRegion_, [&] ( VertId v )
    {
        const float r = std::clamp( editingDistanceMap_[v] / settings_.radius, 0.f, 1.f );
        const float k = r < intensity ? a1 * r * r + 1 : a2 * ( r - 1 ) * ( r - 1 ); // I(r)
        float pointShift = maxShift * k; // shift = F * I(r)
        if ( pointShift > pointsShift_[v] )
        {
            pointShift -= pointsShift_[v];
            pointsShift_[v] += pointShift;
        }
        else
            return;
        points[v] += direction * pointShift * normal;
    } );
    generalEditingRegion_ |= singleEditingRegion_;
    obj_->setDirtyFlags( DIRTY_PRIMITIVES );
}

void SurfaceManipulationWidget::updateUV_( bool set )
{
    BitSetParallelFor( visualizationRegion_, [&] ( VertId v )
    {
        uvs_[v] = set ? UVCoord{ 0, std::clamp( visualizationDistanceMap_[v] / settings_.radius / 2.f, 0.f, 1.f ) } : UVCoord{ 0, 1 };
    } );
}

void SurfaceManipulationWidget::updateRegion_( const Vector2f& mousePos )
{
    MR_TIMER;

    const auto& viewerRef = getViewerInstance();
    std::vector<ObjAndPick> movedPosPick;
    auto objMeshPtr = oldMesh_ ? oldMesh_ : obj_;

    std::vector<Vector2f> viewportPoints;
    if ( mousePos == mousePos_ )
        viewportPoints.push_back( Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewerRef.getHoveredViewportId() ) ) );
    else
    {
        const Vector2f newMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewerRef.getHoveredViewportId() ) );
        const Vector2f oldMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos_ ), viewerRef.getHoveredViewportId() ) );
        const Vector2f vec = newMousePos - oldMousePos;
        const int count = int( std::ceil( vec.length() ) ) + 1;
        const Vector2f step = vec / ( count - 1.f );
        viewportPoints.resize( count );
        for ( int i = 0; i < count; ++i )
            viewportPoints[i] = oldMousePos + step * float( i );
    }
    movedPosPick = getViewerInstance().viewport().multiPickObjects( { objMeshPtr.get() }, viewportPoints );
    mousePos_ = mousePos;

    const auto& mesh = *obj_->mesh();
    std::vector<MeshTriPoint> triPoints;
    VertBitSet newVerts( singleEditingRegion_.size() );
    triPoints.reserve( movedPosPick.size() );
    for ( int i = 0; i < movedPosPick.size(); ++i )
    {
        if ( movedPosPick[i].first == objMeshPtr )
        {
            const auto& pick = movedPosPick[i].second;
            VertId v[3];
            mesh.topology.getTriVerts( pick.face, v );
            for ( int j = 0; j < 3; ++j )
                newVerts.set( v[j] );
            triPoints.push_back( mesh.toTriPoint( pick.face, pick.point ) );
        }
    }

    updateUV_( false );
    const ObjAndPick& curentPosPick = movedPosPick.back();
    visualizationRegion_.reset();
    if ( curentPosPick.first == objMeshPtr )
    {
        visualizationDistanceMap_ = computeSpaceDistances( mesh, { curentPosPick.second.face,curentPosPick.second.point }, settings_.radius * 1.5f );
        VertId v[3];
        mesh.topology.getTriVerts( curentPosPick.second.face, v );
        for ( int j = 0; j < 3; ++j )
            visualizationRegion_.set( v[j] );
        dilateRegion( mesh, visualizationRegion_, settings_.radius * 1.5f );
        updateUV_( true );
    }
    obj_->setAncillaryUVCoords( uvs_ );

    singleEditingRegion_ = newVerts;
    dilateRegion( mesh, singleEditingRegion_, settings_.radius * 1.5f );
    if ( triPoints.size() == 1 )
        editingDistanceMap_ = computeSpaceDistances( mesh, { mesh.topology.left( triPoints[0].e ), mesh.triPoint( triPoints[0] ) }, settings_.radius * 1.5f );
    else
        editingDistanceMap_ = computeSurfaceDistances( mesh, triPoints, settings_.radius * 1.5f, &singleEditingRegion_ );

    for ( auto v : singleEditingRegion_ )
        singleEditingRegion_.set( v, editingDistanceMap_[v] <= settings_.radius );
}

}
