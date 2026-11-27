#include "MRRayWidget.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRSceneRoot.h"
#include <MRMesh/MRMakeSphereMesh.h>
#include <GLFW/glfw3.h>

namespace MR
{

void RayWidget::create( const Line3f& ray, float length, OnRayChangedCallback onRayChanged )
{
    line_ = ray;
    length_ = length;
    onRayChanged_ = onRayChanged;
    pickingPoint_ = false;
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
    MouseClickListener::disconnect(); // Used in pick mode only
    updateRay( ray );
}

void RayWidget::reset()
{
    clearObject_();
    disconnect();
    setPickMode( false );
    onRayChanged_ = {};
}

void RayWidget::updateRay( const Line3f& line )
{
    if ( line.d == Vector3f() )
        return;
    line_.p = line.p;
    line_.d = line.d.normalized();
    if ( !lineObj_ )
        createObject_();
    updateXf_();
}

void RayWidget::updateDirection( const Vector3f& dir )
{
    if ( dir == Vector3f() )
        return;
    line_.d = dir.normalized();
    if ( !lineObj_ )
        createObject_();
    updateXf_();
}

void RayWidget::updateOrigin( const Vector3f& org )
{
    line_.p = org;
    if ( !lineObj_ )
    {
        createObject_();
        updateXf_();
        return;
    }
    auto xf = lineObj_->xf();
    xf.b = line_.p;
    lineObj_->setXf( xf );
}

void RayWidget::updateLength( float length )
{
    if ( length <= 0.0f )
        return;
    length_ = length;
    createObject_();
    updateXf_();
}

void RayWidget::setVisible( bool visible )
{
    if ( lineObj_ )
        lineObj_->setVisible( visible );
}

void RayWidget::setPickMode( bool pickMode )
{
    if ( pickingPoint_ == pickMode )
        return;
    pickingPoint_ = pickMode;
    if ( pickingPoint_ )
        MouseClickListener::connect( &getViewerInstance(), 10, boost::signals2::at_front );
    else
        MouseClickListener::disconnect();
}

void RayWidget::createObject_()
{
    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeCylinder( length_ * 0.01f, length_ ) );
    for ( Vector3f& p : mesh->points )
        p.z -= length_ / 2;
    UndirectedEdgeBitSet creases = mesh->findCreaseEdges( PI_F / 4 );
    cylinderFaces_ = mesh->topology.faceSize();

    mesh->addMesh( makeSphere( { .radius = length_ * 0.03f } ) );
    creases.resize( mesh->topology.undirectedEdgeSize(), false );

    if ( lineObj_ )
        lineObj_->detachFromParent();
    lineObj_ = std::make_shared<ObjectMesh>();
    lineObj_->setAncillary( true );
    lineObj_->setFrontColor( Color::green(), false );
    lineObj_->setMesh( mesh );
    lineObj_->setCreases( creases );
    SceneRoot::get().addChild( lineObj_ );
}

void RayWidget::clearObject_()
{
    if ( lineObj_ )
    {
        lineObj_->detachFromParent();
        lineObj_.reset();
    }
}

void RayWidget::updateXf_()
{
    lineObj_->setXf( AffineXf3f( Matrix3f::rotation( Vector3f::plusZ(), line_.d ), line_.p ) );
}

bool RayWidget::onDragStart_( Viewer::MouseButton button, int modifiers )
{
    pressed_ = false; // even if a drag end was missed
    if ( button != Viewer::MouseButton::Left || modifiers != 0 )
        return false;

    auto viewer = Viewer::instance();
    viewer->select_hovered_viewport();
    const auto [obj, pof] = viewer->viewport().pick_render_object();
    if ( !obj || obj != lineObj_ )
        return false;

    settingDir_ = pof.face < FaceId( cylinderFaces_ );
    worldStartPoint_ = lineObj_->worldXf()( pof.point );
    worldStartOrg_ = line_.p;
    Vector3f projectedStartPoint = line_.project( worldStartPoint_ );
    startDirRotation_ = Matrix3f::rotation( worldStartPoint_ - worldStartOrg_, projectedStartPoint - worldStartOrg_ );
    viewportStartPointZ_ = viewer->viewport().projectToViewportSpace( worldStartPoint_ ).z;
    pressed_ = true;
    return true;
}

bool RayWidget::onDragEnd_( Viewer::MouseButton, int )
{
    return std::exchange( pressed_, false );
}

bool RayWidget::onDrag_( int x, int y )
{
    if ( !pressed_ )
        return false;
    Viewer * viewer = Viewer::instance();
    Vector3f viewportEnd = viewer->screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewer->viewport().id );
    Vector3f worldEndPoint = viewer->viewport().unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );
    if ( settingDir_ )
    {
        Vector3f newDir = startDirRotation_ * ( worldEndPoint - line_.p );
        if ( newDir.lengthSq() > length_ * length_ * 0.0001f )
            updateDirection( newDir );
    }
    else
    {
        Vector3f newOrg = worldStartOrg_ + worldEndPoint - worldStartPoint_;
        updateOrigin( newOrg );
    }
    if ( onRayChanged_ )
        onRayChanged_( line_ );
    return true;
}

bool RayWidget::onMouseClick_( MouseButton button, int modifiers )
{
    Viewer& viewer = getViewerInstance();
    viewer.select_hovered_viewport();
    if ( button == MouseButton::Left && ( modifiers == 0 || modifiers == GLFW_MOD_SHIFT ) )
    {
        auto [pickedObj, pick] = viewer.viewport().pick_render_object();
        if ( pickedObj && !pickedObj->isAncillary() )
        {
            Vector3f pt = pickedObj->worldXf()( pick.point );
            if ( modifiers == 0 )
                updateOrigin( pt );
            else
                updateDirection( pt - line_.p );
            setPickMode( false );
            if ( onRayChanged_ )
                onRayChanged_( line_ );
            return true;
        }
    }
    return false;
}

}
