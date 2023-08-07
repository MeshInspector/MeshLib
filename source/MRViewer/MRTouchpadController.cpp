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

    initRotateParams_ = viewer.viewport().getParameters();
    viewer.eventQueue.emplace( { "Rotation touchpad gesture started", [this]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        // rotate camera around the scene's center
        viewport.rotationCenterMode( Viewport::Parameters::RotationCenterMode::Static );
        viewport.setRotation( true );
        viewport.rotationCenterMode( initRotateParams_.rotationMode );
    } } );

    return rotateChange_( angle );
}

bool TouchpadController::rotateChange_( float angle )
{
    auto& viewer = getViewerInstance();

    viewer.eventQueue.emplace( { "Rotation touchpad gesture updated", [this, angle]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        const auto rot = Matrix3f::rotation( Vector3f::plusZ(), angle );
        viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle * Quaternionf( rot ) );
    } }, true );

    return true;
}

bool TouchpadController::rotateCancel_()
{
    auto& viewer = getViewerInstance();

    viewer.eventQueue.emplace( { "Rotation touchpad gesture canceled", [this]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.setCameraTrackballAngle( initRotateParams_.cameraTrackballAngle );
    } } );

    return rotateEnd_();
}

bool TouchpadController::rotateEnd_()
{
    auto& viewer = getViewerInstance();

    viewer.eventQueue.emplace( { "Rotation touchpad gesture ended", []
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.setRotation( false );
    } } );

    return true;
}

bool TouchpadController::swipe_( float deltaX, float deltaY, bool kinetic )
{
    if ( parameters_.ignoreKineticMoves && kinetic )
        return true;

    auto& viewer = getViewerInstance();
    const auto& viewport = viewer.viewport();

    Vector3f sceneCenterPos;
    if ( viewport.getSceneBox().valid() )
        sceneCenterPos = viewport.getSceneBox().center();
    const auto sceneCenterVpPos = viewport.projectToViewportSpace( sceneCenterPos );

    const auto mousePos = viewer.mouseController.getMousePos();
    const auto oldScreenPos = Vector3f( (float)mousePos.x, (float)mousePos.y, sceneCenterVpPos.z );
    const auto newScreenPos = oldScreenPos + parameters_.swipeScale * Vector3f( deltaX, deltaY, 0.f );

    const auto oldVpPos = viewer.screenToViewport( oldScreenPos, viewport.id );
    const auto newVpPos = viewer.screenToViewport( newScreenPos, viewport.id );

    const auto oldWorldPos = viewport.unprojectFromViewportSpace( oldVpPos );
    const auto newWorldPos = viewport.unprojectFromViewportSpace( newVpPos );

    const auto xf = AffineXf3f::translation( newWorldPos - oldWorldPos );
    viewer.eventQueue.emplace( { "Swipe touchpad gesture", [xf]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.transformView( xf );
    } } );

    return true;
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

    viewer.eventQueue.emplace( { "Zoom touchpad gesture updated", [this, scale]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        constexpr float minAngle = 0.001f;
        constexpr float maxAngle = 179.99f;
        viewport.setCameraViewAngle( std::clamp( initZoomParams_.cameraViewAngle * scale, minAngle, maxAngle ) );
    } }, true );

    return true;
}

bool TouchpadController::zoomCancel_()
{
    auto& viewer = getViewerInstance();

    viewer.eventQueue.emplace( { "Zoom touchpad gesture canceled", [this]
    {
        auto& viewer = getViewerInstance();
        auto& viewport = viewer.viewport();

        viewport.setCameraViewAngle( initZoomParams_.cameraViewAngle );
    } } );

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
    auto& viewer = getViewerInstance();
    viewer.mouseScroll( dy );
}

void TouchpadController::Handler::rotate( float angle, GestureState state )
{
    auto& viewer = getViewerInstance();
    switch ( state )
    {
        case GestureState::Begin:
            viewer.touchpadRotateStart( angle );
            break;
        case GestureState::Change:
            viewer.touchpadRotateChange( angle );
            break;
        case GestureState::End:
            viewer.touchpadRotateEnd();
            break;
        case GestureState::Cancel:
            viewer.touchpadRotateCancel();
            break;
    }
}

void TouchpadController::Handler::swipe( float dx, float dy, bool kinetic )
{
    auto& viewer = getViewerInstance();
    viewer.touchpadSwipe( dx, dy, kinetic );
}

void TouchpadController::Handler::zoom( float scale, GestureState state )
{
    auto& viewer = getViewerInstance();
    switch ( state )
    {
        case GestureState::Begin:
            viewer.touchpadZoomStart( scale );
            break;
        case GestureState::Change:
            viewer.touchpadZoomChange( scale );
            break;
        case GestureState::End:
            viewer.touchpadZoomEnd();
            break;
        case GestureState::Cancel:
            viewer.touchpadZoomCancel();
            break;
    }
}

}