#include "MRTouchpadController.h"
#include "MRViewer.h"
#include "MRMouseController.h"

#if defined( __APPLE__ )
#include "MRTouchpadCocoaHandler.h"
#elif defined( _WIN32 )
#include "MRTouchpadWin32Handler.h"
#endif

#include <GLFW/glfw3.h>

namespace MR
{

void TouchpadController::initialize( GLFWwindow* window )
{
#if defined( __APPLE__ )
    handler_ = std::make_unique<TouchpadCocoaHandler>( window );
#elif defined( _WIN32 )
    handler_ = std::make_unique<TouchpadWin32Handler>( window );
#else
    (void)window;
#endif
}

void TouchpadController::reset()
{
    handler_.reset();
}

bool TouchpadController::touchpadRotateGestureBegin_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    initRotateParams_ = viewer.viewport().getParameters();

    // rotate camera around the scene's center
    viewport.rotationCenterMode( Viewport::Parameters::RotationCenterMode::Static );
    viewport.setRotation( true );

    return true;
}

bool TouchpadController::touchpadRotateGestureUpdate_( float angle )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const auto rot = Matrix3f::rotation( Vector3f::plusZ(), angle );
    viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle * Quaternionf( rot ) );

    return true;
}

bool TouchpadController::touchpadRotateGestureEnd_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    viewport.setRotation( false );
    // restore initial rotation center mode
    viewport.rotationCenterMode( initRotateParams_.rotationMode );

    return true;
}

bool TouchpadController::touchpadSwipeGestureBegin_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    currentSwipeMode_ = parameters_.swipeMode;
    if ( ImGui::GetIO().KeyAlt )
    {
        switch ( parameters_.swipeMode )
        {
        case TouchpadParameters::SwipeMode::SwipeRotatesCamera:
            currentSwipeMode_ = TouchpadParameters::SwipeMode::SwipeMovesCamera;
            break;
        case TouchpadParameters::SwipeMode::SwipeMovesCamera:
            currentSwipeMode_ = TouchpadParameters::SwipeMode::SwipeRotatesCamera;
            break;
        case TouchpadParameters::SwipeMode::Count:
            break;
        }
    }

    if ( currentSwipeMode_ == TouchpadParameters::SwipeMode::SwipeRotatesCamera )
    {
        viewport.setRotation( true );
    }

    return true;
}

bool TouchpadController::touchpadSwipeGestureUpdate_( float deltaX, float deltaY, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return true;

    const auto swipeDirection = Vector3f( deltaX, deltaY, 0.f );

    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    Vector3f sceneCenterPos;
    if ( viewport.getSceneBox().valid() )
        sceneCenterPos = viewport.getSceneBox().center();

    switch ( currentSwipeMode_ )
    {
    case TouchpadParameters::SwipeMode::SwipeRotatesCamera:
    {
        auto quat = viewport.getParameters().cameraTrackballAngle;
        const auto maxDim = (float)std::max( viewer.framebufferSize.x, viewer.framebufferSize.y );
        const auto angle = 4.f * PI_F * swipeDirection / maxDim;
        quat = (
            quat.inverse()
            * Quaternionf( Vector3f::plusY(), angle.x )
            * Quaternionf( Vector3f::plusX(), angle.y )
            * quat
        ).normalized();
        const auto xf = AffineXf3f::linear( Matrix3f( quat ) );
        viewport.transformView( xf );

        return true;
    }
    case TouchpadParameters::SwipeMode::SwipeMovesCamera:
    {
        // do match RMB translation behavior: use the same reference depth as in the MouseController
        constexpr float zpos = 0.75f;

        const Vector2i& mousePos = viewer.mouseController().getMousePos();
        const Vector2f windowDelta{ deltaX, deltaY };
        const Vector2f framebufferDelta = windowDelta * viewer.pixelRatio;
        const Vector3f oldScreenPos = Vector3f( ( float )mousePos.x, ( float )mousePos.y, zpos );
        const Vector3f newScreenPos = oldScreenPos + Vector3f( framebufferDelta.x, framebufferDelta.y, 0.f );

        const Vector3f oldVpPos = viewer.screenToViewport( oldScreenPos, viewport.id );
        const Vector3f newVpPos = viewer.screenToViewport( newScreenPos, viewport.id );

        const Vector3f oldWorldPos = viewport.unprojectFromViewportSpace( oldVpPos );
        const Vector3f newWorldPos = viewport.unprojectFromViewportSpace( newVpPos );

        const auto xf = AffineXf3f::translation( newWorldPos - oldWorldPos );
        viewport.transformView( xf );

        // keep GLFW cursor in sync with the logical mouse so switching to RMB translation does not jump
        Vector2d pos;
        glfwGetCursorPos( viewer.window, &pos.x, &pos.y );
        pos += Vector2d( ( double )windowDelta.x, ( double )windowDelta.y );
        glfwSetCursorPos( viewer.window, pos.x, pos.y );
#if defined( __APPLE__ )
        // on macOS glfwSetCursorPos may not immediately emit a mouse move; update internal state to avoid a jump
        viewer.mouseMove( int( std::round( pos.x * viewer.pixelRatio ) ), int( std::round( pos.y * viewer.pixelRatio ) ) );
#endif

        return true;
    }
    case TouchpadParameters::SwipeMode::Count:
        break;
    }

    MR_UNREACHABLE
}

bool TouchpadController::touchpadSwipeGestureEnd_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    if ( currentSwipeMode_ == TouchpadParameters::SwipeMode::SwipeRotatesCamera )
    {
        viewport.setRotation( false );
    }

    return true;
}

bool TouchpadController::touchpadZoomGestureBegin_()
{
    auto& viewer = getViewerInstance();
    initZoomParams_ = viewer.viewport().getParameters();

    return true;
}

bool TouchpadController::touchpadZoomGestureUpdate_( float scale, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return true;

    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const auto currentViewAngle = viewport.getParameters().cameraViewAngle;

    constexpr float minAngle = 0.001f;
    constexpr float maxAngle = 179.99f;
    // more natural zoom scale
    const auto viewAngle = std::exp( 1.f - scale ) * initZoomParams_.cameraViewAngle;

    const auto mult = std::clamp( viewAngle, minAngle, maxAngle ) / currentViewAngle;
    const auto delta2 = std::log( mult ) / std::log( 0.95f );
    const auto sign = delta2 >= 0.f ? +1.f : -1.f;
    const auto delta = sign * std::sqrt( std::abs( delta2 ) );

    viewer.mouseScroll( delta );

    return true;
}

bool TouchpadController::touchpadZoomGestureEnd_()
{
    return true;
}

const TouchpadParameters& TouchpadController::getParameters() const
{
    return parameters_;
}

void TouchpadController::setParameters( const TouchpadParameters& parameters )
{
    parameters_ = parameters;
}

void TouchpadController::Handler::mouseScroll( float, float dy, bool )
{
    ENQUEUE_VIEWER_METHOD_ARGS( "Mouse scroll", mouseScroll, dy );
}

void TouchpadController::Handler::rotate( float angle, GestureState state )
{
    switch ( state )
    {
        case GestureState::Begin:
            ENQUEUE_VIEWER_METHOD( "Rotation touchpad gesture started", touchpadRotateGestureBegin );
            break;
        case GestureState::Update:
            ENQUEUE_VIEWER_METHOD_ARGS_SKIPABLE( "Rotation touchpad gesture updated", touchpadRotateGestureUpdate, angle );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD( "Rotation touchpad gesture ended", touchpadRotateGestureEnd );
            break;
    }
}

void TouchpadController::Handler::swipe( float dx, float dy, bool kinetic, GestureState state )
{
    switch ( state )
    {
        case GestureState::Begin:
            ENQUEUE_VIEWER_METHOD( "Swipe touchpad gesture started", touchpadSwipeGestureBegin );
            break;
        case GestureState::Update:
            ENQUEUE_VIEWER_METHOD_ARGS_SKIPABLE( "Swipe touchpad gesture updated", touchpadSwipeGestureUpdate, dx, dy, kinetic );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD( "Swipe touchpad gesture ended", touchpadSwipeGestureEnd );
            break;
    }
}

void TouchpadController::Handler::zoom( float scale, bool kinetic, GestureState state )
{
    switch ( state )
    {
        case GestureState::Begin:
            ENQUEUE_VIEWER_METHOD( "Zoom touchpad gesture started", touchpadZoomGestureBegin );
            break;
        case GestureState::Update:
            ENQUEUE_VIEWER_METHOD_ARGS_SKIPABLE( "Zoom touchpad gesture updated", touchpadZoomGestureUpdate, scale, kinetic );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD( "Zoom touchpad gesture ended", touchpadZoomGestureEnd );
            break;
    }
}

}
