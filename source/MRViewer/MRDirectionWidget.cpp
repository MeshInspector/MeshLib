#include "MRDirectionWidget.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRLine.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRArrow.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MR2to3.h"

namespace MR
{

void DirectionWidget::create( Object* parent )
{
    reset();
    connect( &getViewerInstance(), 10, boost::signals2::at_front );

    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>( makeArrow( {}, Vector3f::plusZ(), 0.02f, 0.04f, 0.08f ) );
    directionObj_ = std::make_shared<ObjectMesh>();
    directionObj_->setMesh( mesh );
    directionObj_->setAncillary( true );
    directionObj_->setFrontColor( color_, false );
    directionObj_->setFlatShading( true );

    if ( !parent )
        parent = &SceneRoot::get();
    parent->addChild( directionObj_ );
}

void DirectionWidget::create( const Vector3f& worldDir, const Vector3f& worldBase, float worldLength, OnDirectionChangedCallback onDirectionChanged, Object* parent )
{
    onDirectionChanged_ = onDirectionChanged;
    create( parent );
    updateArrow( { .dir = worldDir, .base = worldBase, .length = worldLength } );
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

void DirectionWidget::updateLocalArrow( const Arrow& arrow )
{
    if ( !directionObj_ )
        return assert( false );

    directionObj_->setXf( AffineXf3f( arrow.length * Matrix3f::rotation( Vector3f::plusZ(), arrow.dir ), arrow.base ) );
}

void DirectionWidget::updateArrow( const Arrow& arrow )
{
    if ( !directionObj_ )
        return assert( false );

    directionObj_->setWorldXf( AffineXf3f( arrow.length * Matrix3f::rotation( Vector3f::plusZ(), arrow.dir ), arrow.base ) );
}

void DirectionWidget::updateLocalDirection( const Vector3f& dir )
{
    auto arrow = getLocalArrow();
    arrow.dir = dir;
    updateLocalArrow( arrow );
}

void DirectionWidget::updateDirection( const Vector3f& dir )
{
    auto arrow = getArrow();
    arrow.dir = dir;
    updateArrow( arrow );
}

void DirectionWidget::updateLocalBase( const Vector3f& base )
{
    if ( !directionObj_ )
        return assert( false );

    auto xf = directionObj_->xf();
    xf.b = base;
    directionObj_->setXf( xf );
}

void DirectionWidget::updateBase( const Vector3f& base )
{
    if ( !directionObj_ )
        return assert( false );
    auto parent = directionObj_->parent();
    if ( !parent )
        return assert( false );

    updateLocalBase( parent->worldXf().inverse()( base ) );
}

void DirectionWidget::updateLocalLength( float length )
{
    auto arrow = getLocalArrow();
    arrow.length = length;
    updateLocalArrow( arrow );
}

void DirectionWidget::updateLength( float length )
{
    auto arrow = getArrow();
    arrow.length = length;
    updateArrow( arrow );
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
    // Get picked point and corresponding arrow axis point
    Vector3f worldStartPoint = directionObj_->worldXf()( pof.point );
    Vector3f projectedStartPoint = Line3f( getBase(), getDirection() ).project( worldStartPoint );
    // Set offset so that worldStartPoint corresponds to projectedStartPoint
    draggingViewportStartPoint_ = viewer->viewport().projectToViewportSpace( worldStartPoint );
    draggingViewportStartPointOffset_ = viewer->viewport().projectToViewportSpace( projectedStartPoint ) - draggingViewportStartPoint_;

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
    const Vector3f viewportCurrentPoint = viewer->screenToViewport(
        Vector3f( float( x ), float( y ), draggingViewportStartPoint_.z ), // Z is preserved by screenToViewport
        viewer->viewport().id );
    const Vector3f worldAxisPoint = viewer->viewport().unprojectFromViewportSpace(
        viewportCurrentPoint + draggingViewportStartPointOffset_ );
    const Vector3f newDir = worldAxisPoint - getBase();
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

Vector3f DirectionWidget::getLocalBase() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return directionObj_->xf().b;
}

Vector3f DirectionWidget::getBase() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return directionObj_->worldXf().b;
}

Vector3f DirectionWidget::getLocalDirection() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return ( directionObj_->xf().A * Vector3f::plusZ() ).normalized();
}

Vector3f DirectionWidget::getDirection() const
{
    if ( !directionObj_ )
        return assert( false ), Vector3f{};

    return ( directionObj_->worldXf().A * Vector3f::plusZ() ).normalized();
}

float DirectionWidget::getLocalLength() const
{
    if ( !directionObj_ )
        return assert( false ), 0.0f;

    return ( directionObj_->xf().A * Vector3f::plusZ() ).length();
}

float DirectionWidget::getLength() const
{
    if ( !directionObj_ )
        return assert( false ), 0.0f;

    return ( directionObj_->worldXf().A * Vector3f::plusZ() ).length();
}

auto DirectionWidget::getLocalArrow() const -> Arrow
{
    Arrow res;
    if ( !directionObj_ )
        return assert( false ), res;

    const auto xf = directionObj_->xf();
    const auto arrowVec = xf.A * Vector3f::plusZ();
    res.length = arrowVec.length();
    if ( res.length )
        res.dir = arrowVec / res.length;
    res.base = xf.b;
    return res;
}

auto DirectionWidget::getArrow() const -> Arrow
{
    Arrow res;
    if ( !directionObj_ )
        return assert( false ), res;

    const auto xf = directionObj_->worldXf();
    const auto arrowVec = xf.A * Vector3f::plusZ();
    res.length = arrowVec.length();
    if ( res.length )
        res.dir = arrowVec / res.length;
    res.base = xf.b;
    return res;
}

Object* DirectionWidget::getParentPtr() const
{
    return directionObj_ ? directionObj_->parent() : nullptr;
}

} // namespace MR
