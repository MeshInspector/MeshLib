#include "MRPlaneWidget.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRMakePlane.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPlaneObject.h"
#include "imgui_internal.h"

namespace MR
{

void PlaneWidget::updatePlane( const Plane3f& plane, bool updateCameraRotation )
{
    plane_ = plane;
    updateWidget_( updateCameraRotation );
    if ( onPlaneUpdate_ )
        onPlaneUpdate_();
}

void PlaneWidget::updateBox( const Box3f& box, bool updateCameraRotation )
{
    box_ = box;
    if ( planeObj_ )
        updateWidget_( updateCameraRotation );
}

void PlaneWidget::definePlane()
{
    if ( planeObj_ )
        return;

    std::shared_ptr<Mesh> planeMesh = std::make_shared<Mesh>( makePlane() );
    planeObj_ = std::make_shared<ObjectMesh>();
    planeObj_->setName( "PlaneObject" );
    planeObj_->setMesh( planeMesh );
    planeObj_->setAncillary( true );
    planeObj_->setVisualizeProperty( true, MeshVisualizePropertyType::OnlyOddFragments, ViewportMask::all() );
    planeObj_->setBordersColor( SceneColors::get( SceneColors::Type::Labels ) );
    planeObj_->setVisualizeProperty( true, MeshVisualizePropertyType::BordersHighlight, ViewportMask::all() );
    SceneRoot::get().addChild( planeObj_ );

    updateWidget_();
}

void PlaneWidget::undefinePlane()
{
    if ( !planeObj_ )
        return;
    
    planeObj_->detachFromParent();
    planeObj_.reset();    
}

const Plane3f& PlaneWidget::getPlane() const
{
    return plane_;
}

const std::shared_ptr<ObjectMesh>& PlaneWidget::getPlaneObject() const
{
    return planeObj_;
}

void PlaneWidget::setOnPlaneUpdateCallback( OnPlaneUpdateCallback callback )
{
    onPlaneUpdate_ = callback;
}

const Box3f& PlaneWidget::box() const
{
    return box_;
}

bool PlaneWidget::importPlaneMode() const
{
    return importPlaneMode_;
}

void PlaneWidget::setImportPlaneMode( bool val )
{
    importPlaneMode_ = val;
}

void PlaneWidget::updateWidget_( bool updateCameraRotation )
{
    if ( !planeObj_ )
        definePlane();

    auto viewer = Viewer::instance();
    plane_ = plane_.normalized();

    auto trans1 = AffineXf3f::translation( plane_.project( box_.center() ) );
    auto rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), plane_.n ) );
    auto scale1 = AffineXf3f::linear( Matrix3f::scale( box_.diagonal() ) );
    AffineXf3f transform = trans1 * rot1 * scale1;
    if ( updateCameraRotation )
        cameraUp3Old_ = viewer->viewport().getUpDirection();
    Vector3f cameraUp3 = cameraUp3Old_;
    auto from = transform.A * Vector3f::plusY();
    auto to = plane_.project( transform.b + cameraUp3 ) - transform.b;
    auto rot2 = Matrix3f::rotation( plane_.n, angle( from, to ) );

    auto lastPlaneTransform = trans1 * AffineXf3f::linear( rot2 ) * rot1;
    transform = lastPlaneTransform * scale1;
    planeObj_->setXf( transform );
}

bool PlaneWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != Viewer::MouseButton::Left || mod != 0 )
        return false;

    auto viewer = Viewer::instance();

    if ( importPlaneMode_ )
    {
        const auto [obj, point] = viewer->viewport().pick_render_object();
        if ( !obj )
            return false;
        auto planeObj = std::dynamic_pointer_cast< PlaneObject >( obj );
        if ( !planeObj )
            return false;

        plane_ = Plane3f::fromDirAndPt( planeObj->getNormal(), planeObj->getCenter() );
        definePlane();
        updatePlane( plane_ );

        importPlaneMode_ = false;
        return true;
    }

    
    const auto& mousePos = viewer->mouseController.getMousePos();
    startMousePos_ = endMousePos_ = Vector2f( float ( mousePos.x ), float ( mousePos.y ) );
    pressed_ = true;

    if ( line_ )
    {
        line_->detachFromParent();
        line_.reset();
    }

    line_ = std::make_shared<ObjectLines>();
    line_->setName( "PlaneLine" );
    line_->setAncillary( true );
    
    const auto lineColor = SceneColors::get( SceneColors::Type::Labels );
    line_->setFrontColor( lineColor, false );
    line_->setBackColor( lineColor );

    auto currentViewportId = viewer->viewport().id;
    line_->setVisualizeProperty( false, VisualizeMaskType::DepthTest, currentViewportId );
    line_->setVisibilityMask( currentViewportId );

    SceneRoot::get().addChild( line_ );

    return true;
}

bool PlaneWidget::onMouseUp_( Viewer::MouseButton, int )
{
    if ( !pressed_ )
        return false;

    line_->detachFromParent();
    line_.reset();

    pressed_ = false;

    ImVec2 dir;
    dir.x = endMousePos_.x - startMousePos_.x;
    dir.y = endMousePos_.y - startMousePos_.y;
    if ( ( endMousePos_ - startMousePos_ ).lengthSq() < 50 * 50 )
        return false;

    auto viewer = Viewer::instance();
    auto& viewport = viewer->viewport();
    auto viewportStart = viewer->screenToViewport( to3dim( startMousePos_ ), viewport.id );
    auto start = viewport.unprojectFromViewportSpace( viewportStart );

    auto viewportStop = viewer->screenToViewport( to3dim( endMousePos_ ), viewport.id );
    auto stop = viewport.unprojectFromViewportSpace( viewportStop );
    auto stopFar = viewport.unprojectFromViewportSpace( { viewportStop.x, viewportStop.y, 1.0f } );

    auto prevNorm = plane_.n;
    plane_ = Plane3f::fromDirAndPt( cross( stopFar - stop, stop - start ).normalized(), start );
    if ( angle( -plane_.n, prevNorm ) < angle( plane_.n, prevNorm ) )
        plane_ = -plane_;
    updatePlane( plane_ );
    return true;
}

bool PlaneWidget::onMouseMove_( int mouse_x, int mouse_y )
{
    if ( !pressed_ )
        return false;

    endMousePos_ = Vector2f( ( float )mouse_x, ( float )mouse_y );
    
    auto viewer = Viewer::instance();
    auto& viewport = viewer->viewport();

    const auto viewportOrigin = viewport.projectToViewportSpace( box_.center() );
    const auto screenOrigin = viewer->viewportToScreen( viewportOrigin, viewport.id );

    auto viewportStart = viewer->screenToViewport( { startMousePos_.x, startMousePos_.y, screenOrigin.z }, viewport.id );
    auto start = viewport.unprojectFromViewportSpace( viewportStart );

    auto viewportStop = viewer->screenToViewport(  { endMousePos_.x, endMousePos_.y, screenOrigin.z } , viewport.id );
    auto stop = viewport.unprojectFromViewportSpace( viewportStop );
    const Polyline3 polyline( { { start, stop } } );
   
    line_->setPolyline( std::make_shared<Polyline3>( polyline ) );

    return true;
}

}