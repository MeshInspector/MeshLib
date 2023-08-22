#include "MRTouchpadController.h"
#include "MRViewer.h"

#if defined( __APPLE__ )
#include "MRTouchpadCocoaHandler.h"
#elif defined( _WIN32 )
#include "MRTouchpadWin32Handler.h"
#endif

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

void TouchpadController::touchpadRotateGestureBegin_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    initRotateParams_ = viewer.viewport().getParameters();

    // rotate camera around the scene's center
    viewport.rotationCenterMode( Viewport::Parameters::RotationCenterMode::Static );
    viewport.setRotation( true );
    viewport.rotationCenterMode( initRotateParams_.rotationMode );
}

void TouchpadController::touchpadRotateGestureUpdate_( float angle )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const auto rot = Matrix3f::rotation( Vector3f::plusZ(), angle );
    viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle * Quaternionf( rot ) );
}

void TouchpadController::touchpadRotateGestureEnd_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    viewport.setRotation( false );
}

void TouchpadController::touchpadSwipeGestureBegin_()
{
    //
}

void TouchpadController::touchpadSwipeGestureUpdate_( float deltaX, float deltaY, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return;

    const auto swipeDirection = Vector3f( deltaX, deltaY, 0.f );

    auto swipeMode = parameters_.swipeMode;
    if ( ImGui::GetIO().KeyAlt )
    {
        switch ( swipeMode )
        {
        case Parameters::SwipeRotatesCamera:
            swipeMode = Parameters::SwipeMovesCamera;
            break;
        case Parameters::SwipeMovesCamera:
            swipeMode = Parameters::SwipeRotatesCamera;
            break;
        case Parameters::SwipeModeCount:
            break;
        }
    }

    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    Vector3f sceneCenterPos;
    if ( viewport.getSceneBox().valid() )
        sceneCenterPos = viewport.getSceneBox().center();

    switch ( swipeMode )
    {
    case Parameters::SwipeRotatesCamera:
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
        const auto xf = AffineXf3f::xfAround( Matrix3f( quat ), sceneCenterPos );
        viewport.transformView( xf );

        return;
    }
    case Parameters::SwipeMovesCamera:
    {
        const auto sceneCenterVpPos = viewport.projectToViewportSpace( sceneCenterPos );

        const auto mousePos = viewer.mouseController.getMousePos();
        const auto oldScreenPos = Vector3f( (float)mousePos.x, (float)mousePos.y, sceneCenterVpPos.z );
        const auto newScreenPos = oldScreenPos + swipeDirection;

        const auto oldVpPos = viewer.screenToViewport( oldScreenPos, viewport.id );
        const auto newVpPos = viewer.screenToViewport( newScreenPos, viewport.id );

        const auto oldWorldPos = viewport.unprojectFromViewportSpace( oldVpPos );
        const auto newWorldPos = viewport.unprojectFromViewportSpace( newVpPos );

        const auto xf = AffineXf3f::translation( newWorldPos - oldWorldPos );
        viewport.transformView( xf );

        return;
    }
    case Parameters::SwipeModeCount:
        break;
    }

#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
    assert( false );
#endif
}

void TouchpadController::touchpadSwipeGestureEnd_()
{
    //
}

void TouchpadController::touchpadZoomGestureBegin_()
{
    auto& viewer = getViewerInstance();

    initZoomParams_ = viewer.viewport().getParameters();
}

void TouchpadController::touchpadZoomGestureUpdate_( float scale, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return;

    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    constexpr float minAngle = 0.001f;
    constexpr float maxAngle = 179.99f;
    // more natural zoom scale
    const auto viewAngle = std::exp( 1.f - scale ) * initZoomParams_.cameraViewAngle;
    viewport.setCameraViewAngle( std::clamp( viewAngle, minAngle, maxAngle ) );
}

void TouchpadController::touchpadZoomGestureEnd_()
{
    //
}

const TouchpadController::Parameters& TouchpadController::getParameters() const
{
    return parameters_;
}

void TouchpadController::setParameters( const TouchpadController::Parameters& parameters )
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
        case GestureState::Change:
            ENQUEUE_VIEWER_METHOD_ARGS( "Rotation touchpad gesture updated", touchpadRotateGestureUpdate, angle );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD_ARGS( "Rotation touchpad gesture ended", touchpadRotateGestureEnd );
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
        case GestureState::Change:
            ENQUEUE_VIEWER_METHOD_ARGS( "Swipe touchpad gesture updated", touchpadSwipeGestureUpdate, dx, dy, kinetic );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD_ARGS( "Swipe touchpad gesture ended", touchpadSwipeGestureEnd );
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
        case GestureState::Change:
            ENQUEUE_VIEWER_METHOD_ARGS( "Zoom touchpad gesture updated", touchpadZoomGestureUpdate, scale, kinetic );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD_ARGS( "Zoom touchpad gesture ended", touchpadZoomGestureEnd );
            break;
    }
}

}