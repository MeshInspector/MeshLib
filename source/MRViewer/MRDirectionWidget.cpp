#include "MRDirectionWidget.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRMatrix3Decompose.h"

namespace MR
{

void DirectionWidget::create( const Vector3f& dir, const Vector3f& base, float length, OnDirectionChangedCallback onDirectionChanged, Object* parent )
{
    reset();
    base_ = base;
    length_ = length;
    onDirectionChanged_ = onDirectionChanged;
    connect( &getViewerInstance(), 10, boost::signals2::at_front );

    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, length_ * Vector3f::plusZ(), length_ * 0.02f, length_ * 0.04f, length_ * 0.08f));
    directionObj_ = std::make_shared<ObjectMesh>();
    directionObj_->setMesh( mesh );
    directionObj_->setAncillary( true );
    directionObj_->setFrontColor( color_, false );
    directionObj_->setFlatShading( true );

    if ( !parent )
        parent = &SceneRoot::get();
    parent->addChild( directionObj_ );

    updateDirection( dir );
}

void DirectionWidget::reset()
{
    clear_();
    disconnect();
}


void DirectionWidget::setOnDirectionChangedCallback( OnDirectionChangedCallback cb )
{
    onDirectionChanged_ = cb;
}

void DirectionWidget::updateLocalDirection( const Vector3f& dir )
{
    if ( !directionObj_ )
        return assert( false );

    return directionObj_->setXf( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), dir ) ) );
}

void DirectionWidget::updateDirection( const Vector3f& dir )
{
    if ( !directionObj_ )
        return assert( false );
    auto parent = directionObj_->parent();
    if ( !parent )
        return assert( false );

    updateLocalDirection( parent->worldXf().A.inverse() * dir );
}

void DirectionWidget::updateBase( const Vector3f& base )
{
    if ( !directionObj_ )
        return;

    base_ = base;
    auto xf = directionObj_->xf();
    xf.b = base_;
    directionObj_->setXf( xf );
}

void DirectionWidget::updateLength( float length )
{
    if ( !directionObj_ )
        return;

    length_ = length;
    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, Vector3f::plusZ() * length_, length_ * 0.02f, length_ * 0.04f, length_ * 0.08f ) );
    directionObj_->setMesh( mesh );
}

void DirectionWidget::updateArrow( const Vector3f& base, float length )
{
    updateBase( base );
    updateLength( length );
}

void DirectionWidget::setVisible( bool visible )
{
    if ( directionObj_ )
        directionObj_->setVisible( visible );
}

bool DirectionWidget::isVisible() const
{
    if ( !directionObj_ )
        return false;
    return directionObj_->isVisible();
}

void DirectionWidget::clear_()
{
    if ( directionObj_ )
    {
        directionObj_->detachFromParent();
        directionObj_.reset();
    }

    mousePressed_ = false;
}

bool DirectionWidget::onMouseDown_( Viewer::MouseButton button, int mod )
{
    if ( button != Viewer::MouseButton::Left || mod != 0 || blockedMouse_ )
        return false;

    auto viewer = Viewer::instance();
    viewer->select_hovered_viewport();
    const auto [obj, pof] = viewer->viewport().pick_render_object();
    if ( obj != directionObj_ )
        return false;

    mousePressed_ = true;
    worldStartPoint_ = directionObj_->worldXf()( pof.point );
    viewportStartPointZ_ = viewer->viewport().projectToViewportSpace( worldStartPoint_ ).z;
    return true;
}

bool DirectionWidget::onMouseMove_( int x, int y )
{
    if ( !mousePressed_ )
        return false;
    if ( !directionObj_ )
        return assert( false ), false;
    auto parent = directionObj_->parent();
    if ( !parent )
        return assert( false ), false;

    auto viewer = Viewer::instance();
    const auto viewportEnd = viewer->screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewer->viewport().id );
    const auto worldEndPoint = viewer->viewport().unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );
    const auto newDir = worldEndPoint - parent->worldXf()( base_ );
    updateDirection( newDir );
    if ( onDirectionChanged_ )
        onDirectionChanged_( newDir, needToSaveHistory_ );

    needToSaveHistory_ = false;
    return true;
}

bool DirectionWidget::onMouseUp_( Viewer::MouseButton button, int )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    if ( !mousePressed_ )
        return false;

    mousePressed_ = false;
    needToSaveHistory_ = true;
    return true;
}

void DirectionWidget::setColor( const Color& color )
{
    color_ = color;
    if ( directionObj_ )
        directionObj_->setFrontColor( color_, false );
}

const Color& DirectionWidget::getColor() const
{
    return color_;
}

const Vector3f& DirectionWidget::getBase() const
{
    return base_;
}

Vector3f DirectionWidget::getLocalDirection() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return directionObj_->xf().A * Vector3f::plusZ();
}

Vector3f DirectionWidget::getDirection() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return directionObj_->worldXf().A * Vector3f::plusZ();
}

Object* DirectionWidget::getParentPtr() const
{
    return directionObj_ ? directionObj_->parent() : nullptr;
}

} // namespace MR
