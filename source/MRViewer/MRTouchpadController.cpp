#include "MRTouchpadController.h"
#include "MRTouchpadCocoaHandler.h"
#include "MRViewer.h"

namespace MR
{

void TouchpadController::initialize( GLFWwindow* window )
{
#ifdef __APPLE__
    handler_ = std::make_unique<TouchpadCocoaHandler>( window );
#else
    (void)window;
#endif
}

void TouchpadController::connect()
{
    auto& viewer = getViewerInstance();
    viewer.touchpadRotateStartSignal.connect( MAKE_SLOT( &TouchpadController::rotateStart_ ) );
    viewer.touchpadRotateChangeSignal.connect( MAKE_SLOT( &TouchpadController::rotateChange_ ) );
    viewer.touchpadRotateEndSignal.connect( MAKE_SLOT( &TouchpadController::rotateEnd_ ) );
    viewer.touchpadRotateCancelSignal.connect( MAKE_SLOT( &TouchpadController::rotateCancel_ ) );
    viewer.touchpadSwipeSignal.connect( MAKE_SLOT( &TouchpadController::swipe_ ) );
    viewer.touchpadZoomStartSignal.connect( MAKE_SLOT( &TouchpadController::zoomStart_ ) );
    viewer.touchpadZoomChangeSignal.connect( MAKE_SLOT( &TouchpadController::zoomChange_ ) );
    viewer.touchpadZoomEndSignal.connect( MAKE_SLOT( &TouchpadController::zoomEnd_ ) );
    viewer.touchpadZoomCancelSignal.connect( MAKE_SLOT( &TouchpadController::zoomCancel_ ) );
}

bool TouchpadController::rotateStart_( float angle )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    initRotateParams_ = viewer.viewport().getParameters();

    // rotate camera around the scene's center
    viewport.rotationCenterMode( Viewport::Parameters::RotationCenterMode::Static );
    viewport.setRotation( true );
    viewport.rotationCenterMode( initRotateParams_.rotationMode );

    return rotateChange_( angle );
}

bool TouchpadController::rotateChange_( float angle )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const auto rot = Matrix3f::rotation( Vector3f::plusZ(), angle );
    viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle * Quaternionf( rot ) );

    return true;
}

bool TouchpadController::rotateCancel_()
{
    if ( parameters_.cancellable )
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle );
    }

    return rotateEnd_();
}

bool TouchpadController::rotateEnd_()
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    viewport.setRotation( false );

    return true;
}

bool TouchpadController::swipe_( float deltaX, float deltaY, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return true;

    const auto swipeDirection = parameters_.swipeScale * Vector3f( deltaX, deltaY, 0.f );

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

        return true;
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

        return true;
    }
    case Parameters::SwipeModeCount:
        break;
    }

#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
    assert( false );
    return {};
#endif
}

bool TouchpadController::zoomStart_( float scale )
{
    auto& viewer = getViewerInstance();

    initZoomParams_ = viewer.viewport().getParameters();

    return zoomChange_( scale );
}

bool TouchpadController::zoomChange_( float scale )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    constexpr float minAngle = 0.001f;
    constexpr float maxAngle = 179.99f;
    // more natural zoom scale
    const auto viewAngle = std::exp( 1.f - scale ) * initZoomParams_.cameraViewAngle;
    viewport.setCameraViewAngle( std::clamp( viewAngle, minAngle, maxAngle ) );

    return true;
}

bool TouchpadController::zoomCancel_()
{
    if ( parameters_.cancellable )
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.setCameraViewAngle( initZoomParams_.cameraViewAngle );
    }

    return zoomEnd_();
}

bool TouchpadController::zoomEnd_()
{
    return true;
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
            ENQUEUE_VIEWER_METHOD_ARGS( "Rotation touchpad gesture started", touchpadRotateStart, angle );
            break;
        case GestureState::Change:
            ENQUEUE_VIEWER_METHOD_ARGS( "Rotation touchpad gesture updated", touchpadRotateChange, angle );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD( "Rotation touchpad gesture ended", touchpadRotateEnd );
            break;
        case GestureState::Cancel:
            ENQUEUE_VIEWER_METHOD( "Rotation touchpad gesture canceled", touchpadRotateCancel );
            break;
    }
}

void TouchpadController::Handler::swipe( float dx, float dy, bool kinetic )
{
    ENQUEUE_VIEWER_METHOD_ARGS( "Swipe touchpad gesture", touchpadSwipe, dx, dy, kinetic );
}

void TouchpadController::Handler::zoom( float scale, GestureState state )
{
    switch ( state )
    {
        case GestureState::Begin:
            ENQUEUE_VIEWER_METHOD_ARGS( "Zoom touchpad gesture started", touchpadZoomStart, scale );
            break;
        case GestureState::Change:
            ENQUEUE_VIEWER_METHOD_ARGS( "Zoom touchpad gesture updated", touchpadZoomChange, scale );
            break;
        case GestureState::End:
            ENQUEUE_VIEWER_METHOD( "Zoom touchpad gesture ended", touchpadZoomEnd );
            break;
        case GestureState::Cancel:
            ENQUEUE_VIEWER_METHOD( "Zoom touchpad gesture canceled", touchpadZoomCancel );
            break;
    }
}

}