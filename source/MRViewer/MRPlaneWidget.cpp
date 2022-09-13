#include "MRPlaneWidget.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRMakePlane.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"
#include "imgui_internal.h"

namespace MR
{

PlaneWidget::PlaneWidget( const Plane3f& plane, const Box3f& box, OnPlaneUpdateCallback onPlaneUpdate )
: plane_(plane),
box_(box),
onPlaneUpdate_( onPlaneUpdate )
{ 
}

PlaneWidget::~PlaneWidget()
{    
    undefinePlane();
}

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
    planeObj_->setFrontColor( Color( Vector4f::diagonal( 0.3f ) ), false );
    planeObj_->setBackColor( Color( Vector4f::diagonal( 0.3f ) ) );
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

void PlaneWidget::setOnPlaneUpdateCalback( OnPlaneUpdateCallback callback )
{
    onPlaneUpdate_ = callback;
}

void PlaneWidget::updateWidget_( bool updateCameraRotation )
{
    auto viewer = Viewer::instance();
    plane_ = plane_.normalized();

    auto trans1 = AffineXf3f::translation( plane_.project( box_.center() ) );
    auto rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), plane_.n ) );
    auto scale1 = AffineXf3f::linear( Matrix3f::scale( box_.diagonal() ) );
    AffineXf3f transform = trans1 * rot1 * scale1;
    if ( updateCameraRotation )
        cameraUp3Old_ = viewer->viewport().getUpDirection();
    Vector3f cameraUp3 = cameraUp3Old_;
    auto rot2 = Matrix3f::rotation( transform.A * Vector3f::plusY(),
                                    plane_.project( transform( Vector3f() ) + cameraUp3 ) - transform( Vector3f() ) );

    auto lastPlaneTransform = trans1 * AffineXf3f::linear( rot2 ) * rot1;
    transform = lastPlaneTransform * scale1;
    planeObj_->setXf( transform );
}

bool PlaneWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != Viewer::MouseButton::Left || mod != 0 )
        return false;

    const auto& mousePos = Viewer::instance()->mouseController.getMousePos();
    startMousePos_ = endMousePos_ = Vector2f( float( mousePos.x ), float( mousePos.y ) );
    pressed_ = true;
    return true;
}

bool PlaneWidget::onMouseUp_( Viewer::MouseButton, int )
{
    if ( !pressed_ )
        return false;

    pressed_ = false;

    ImVec2 dir;
    dir.x = endMousePos_.x - startMousePos_.x;
    dir.y = endMousePos_.y - startMousePos_.y;
    if ( sqr( dir.x ) + sqr( dir.y ) < sqr( 50 ) )
        return false; // if less then 50 pixes were in line

    auto viewer = Viewer::instance();
    auto viewportStart = viewer->screenToViewport( Vector3f( float( startMousePos_.x ), float( startMousePos_.y ), 0.f ), viewer->viewport().id );
    auto start = viewer->viewport().unprojectFromViewportSpace( { viewportStart.x, viewportStart.y, 0.0f } );

    auto viewportStop = viewer->screenToViewport( Vector3f( float( endMousePos_.x ), float( endMousePos_.y ), 0.f ), viewer->viewport().id );
    auto stop = viewer->viewport().unprojectFromViewportSpace( { viewportStop.x, viewportStop.y, 0.0f } );
    auto stopFar = viewer->viewport().unprojectFromViewportSpace( { viewportStop.x, viewportStop.y, 1.0f } );

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
    
    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    window->DrawList->AddLine( ImVec2( startMousePos_.x, startMousePos_.y ), ImVec2( endMousePos_.x, endMousePos_.y ), ImGui::GetColorU32( ImGuiCol_Text ) );

    return true;
}

}