#include "MRMouseController.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRQuaternion.h"
#include "MRMesh/MRVisualObject.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRPch/MRWasm.h"


#ifdef __EMSCRIPTEN__

extern "C"
{
EMSCRIPTEN_KEEPALIVE void emsDropEvents()
{
    auto& viewer = MR::getViewerInstance();
    const auto& ctl = viewer.mouseController;
    if ( ctl.isPressed( MR::MouseButton::Left ) )
        viewer.mouseUp( MR::MouseButton::Left, 0 );
    if ( ctl.isPressed( MR::MouseButton::Right ) )
        viewer.mouseUp( MR::MouseButton::Right, 0 );
    if ( ctl.isPressed( MR::MouseButton::Middle ) )
        viewer.mouseUp( MR::MouseButton::Middle, 0 );
}
}

#endif

namespace MR
{

void MouseController::setMouseControl( const MouseControlKey& key, MouseMode mode )
{
    auto newMapKey = mouseAndModToKey( key );
    auto [backMapIt, insertedToBackMap] = backMap_.insert( { mode,newMapKey } );
    if ( !insertedToBackMap )
    {
        map_.erase( backMapIt->second );
        backMapIt->second = newMapKey;
    }

    auto [mapIt, insertedToMap] = map_.insert( { newMapKey,mode } );
    if ( !insertedToMap )
    {
        backMap_.erase( mapIt->second );
        mapIt->second = mode;
    }
}

bool MouseController::isPressed( MouseButton btn ) const
{
    return downState_.test( int( btn ) );
}

std::optional<MouseController::MouseControlKey> MouseController::findControlByMode( MouseMode mode ) const
{
    auto ctrlIt = backMap_.find( mode );
    if ( ctrlIt == backMap_.end() )
        return {};
    return keyToMouseAndMod( ctrlIt->second );
}

std::string MouseController::getControlString( const MouseControlKey& key )
{
    std::string res;
    if ( key.mod & GLFW_MOD_ALT )
        res += "Alt+";
    if ( key.mod & GLFW_MOD_CONTROL )
        res += "Ctrl+";
    if ( key.mod & GLFW_MOD_SHIFT )
        res += "Shift+";
    switch ( key.btn )
    {
    case MouseButton::Left:
        res += "LMB";
        break;
    case MouseButton::Right:
        res += "RMB";
        break;
    case MouseButton::Middle:
        res += "MMB";
        break;
    default:
        res += "Error";
        break;
    }
    return res;
}

int MouseController::mouseAndModToKey( const MouseControlKey& key )
{
    return int( int( key.btn ) << 6 ) + key.mod;
}

MouseController::MouseControlKey MouseController::keyToMouseAndMod( int key )
{
    return { MouseButton( key >> 6 ),key % ( 1 << 6 ) };
}

void MouseController::setMouseScroll( bool active )
{
    scrollActive_ = active;
}

void MouseController::connect()
{
    downState_.resize( 3 );
    auto& viewer = getViewerInstance();
    viewer.mouseDownSignal.connect( MAKE_SLOT( &MouseController::preMouseDown_ ), boost::signals2::at_front );
    viewer.mouseDownSignal.connect( MAKE_SLOT( &MouseController::mouseDown_ ) );
    viewer.mouseUpSignal.connect( MAKE_SLOT( &MouseController::preMouseUp_ ), boost::signals2::at_front );
    viewer.mouseMoveSignal.connect( MAKE_SLOT( &MouseController::preMouseMove_ ), boost::signals2::at_front );
    viewer.mouseScrollSignal.connect( MAKE_SLOT( &MouseController::mouseScroll_ ) );
    viewer.cursorEntranceSignal.connect( MAKE_SLOT( &MouseController::cursorEntrance_ ) );
}

void MouseController::cursorEntrance_( bool entered )
{
    isCursorInside_ = entered;
}

bool MouseController::preMouseDown_( MouseButton btn, int )
{
    resetAllIfNeeded_();
    if ( !downState_.any() )
        downMousePos_ = currentMousePos_;

    downState_.set( int( btn ) );
    return false;
}

bool MouseController::mouseDown_( MouseButton btn, int mod )
{
    if ( currentMode_ != MouseMode::None )
        return false;

    if ( downState_.count() > 1 )
        return false;

    auto& viewer = getViewerInstance();
    viewer.select_hovered_viewport();

    auto modIt = map_.find( mouseAndModToKey( { btn,mod } ) );
    if ( modIt == map_.end() )
        return false;

    currentMode_ = modIt->second;
    if ( currentMode_ == MouseMode::Rotation || currentMode_ == MouseMode::Roll )
        viewer.viewport().setRotation( true );
    else if ( currentMode_ == MouseMode::Translation )
        downTranslation_ = viewer.viewport().getParameters().cameraTranslation;
    return true;
}

bool MouseController::preMouseUp_( MouseButton btn, int )
{
    downState_.set( int( btn ), false );
    if ( currentMode_ == MouseMode::None )
        return false;

    auto btnIt = backMap_.find( currentMode_ );
    if ( btnIt == backMap_.end() )
        return false;

    if ( keyToMouseAndMod( btnIt->second ).btn != btn )
        return false;

    if ( currentMode_ == MouseMode::Rotation || currentMode_ == MouseMode::Roll )
        getViewerInstance().viewport().setRotation( false );

    currentMode_ = MouseMode::None;
    return false; // so others can override mouse up even if scene control was active
}

bool MouseController::preMouseMove_( int x, int y)
{
    prevMousePos_ = currentMousePos_;
    currentMousePos_ = { x,y };

    if ( currentMode_ == MouseMode::None )
        return false;

    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();
    AffineXf3f xf;
    switch ( currentMode_ )
    {
    case MR::MouseMode::Rotation:
    {
        auto quat = viewport.getParameters().cameraTrackballAngle;
        float maxDimension = float( std::max( viewer.framebufferSize.x, viewer.framebufferSize.y ) );
        auto angle = PI_F * ( Vector2f( currentMousePos_ ) - Vector2f( prevMousePos_ ) ) / maxDimension * 4.0f;
        quat = (
            quat.inverse() *
            Quaternionf( Vector3f{ 0,1,0 }, angle.x ) *
            Quaternionf( Vector3f{ 1,0,0 }, angle.y ) *
            quat
            ).normalized();
        xf = AffineXf3f::linear( Matrix3f( quat ) );
        break;
    }
    case MR::MouseMode::Roll:
    {
        auto quat = viewport.getParameters().cameraTrackballAngle;
        auto angle = PI_F * ( currentMousePos_.x - prevMousePos_.x ) / viewer.framebufferSize.x * 4.0f;
        quat = (
            quat.inverse() *
            Quaternionf( Vector3f{ 0,0,1 }, angle ) *
            quat
            ).normalized();
        xf = AffineXf3f::linear( Matrix3f( quat ) );
        break;
    }
    case MR::MouseMode::Translation:
    {
        //translation
        // const for better perspective translation
        constexpr float zpos = 0.75f;

        auto vpPoint = viewer.screenToViewport( Vector3f( float( currentMousePos_.x ), float( currentMousePos_.y ), zpos ), viewport.id );
        auto vpPointMouseDown = viewer.screenToViewport( Vector3f( float( downMousePos_.x ), float( downMousePos_.y ), zpos ), viewport.id );
        auto pos1 = viewport.unprojectFromViewportSpace( vpPoint );
        auto pos0 = viewport.unprojectFromViewportSpace( vpPointMouseDown );
        xf = AffineXf3f::translation( downTranslation_ + pos1 - pos0 - viewport.getParameters().cameraTranslation );
        break;
    }
    default:
        break;
    }
    if ( transformModifierCb_ )
        transformModifierCb_( xf );
    viewport.transformView( xf );
    return true;
}

bool MouseController::mouseScroll_( float delta )
{
    resetAllIfNeeded_();

    if ( !scrollActive_ )
        return false;

    if ( currentMode_ != MouseMode::None )
        return false;

    if ( delta == 0.0f )
        return false;

    auto& viewer = getViewerInstance();

    viewer.select_hovered_viewport();

    // draw here to update glPixels
    auto& viewport = viewer.viewport();
    auto viewportPoint = viewer.screenToViewport( Vector3f( float( currentMousePos_.x ), float( currentMousePos_.y ), 0.f ), viewport.id );
    auto [obj,pick] = viewport.pick_render_object();
    if ( obj )
    {
        auto worldPoint = obj->worldXf()( pick.point );
        viewportPoint = viewport.projectToViewportSpace( worldPoint );
    }
    else
    {
        // fix z if miss object for perspective
        viewportPoint.z = 0.75f;
    }
    auto ps = viewport.unprojectFromViewportSpace( viewportPoint );
    auto pc = viewport.unprojectFromClipSpace( Vector3f( 0.f, 0.f, viewportPoint.z * 2.f - 1.f ) );

    if ( fabs( delta ) > 4 )  delta = delta / fabs( delta ) * 4;
    float  mult = pow( 0.95f, fabs( delta ) * delta );
    constexpr float min_angle = 0.001f;
    constexpr float max_angle = 179.99f;
    // divide by 360 instead of 180 to have 0.5 rad (maybe)
    constexpr float d2r = PI_F / 360.0f;
    auto newFOV = std::clamp( float( atan( tan( ( viewport.getParameters().cameraViewAngle ) * d2r ) * mult ) / d2r ), min_angle, max_angle );
    if ( fovModifierCb_ )
        fovModifierCb_( newFOV );
    viewport.setCameraViewAngle( newFOV );

    AffineXf3f xf = AffineXf3f::translation( ( ps - pc ) * ( mult - 1.0f ) );
    if ( transformModifierCb_ )
        transformModifierCb_( xf );
    viewport.transformView( xf );
    return true;
}

void MouseController::resetAllIfNeeded_()
{
    if ( !dropOldEventsOnNew_ )
        return;
    for ( auto btn : downState_ )
        getViewerInstance().mouseUp( MouseButton( btn ), 0 );
}

}