#include "MRSpaceMouseController.h"
#include "MRViewer.h"
#include "MRViewerSignals.h"
#include "MRMakeSlot.h"
#include "MRViewerInstance.h"
#include "MRViewport.h"
#include "MRSpaceMouseDevice.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRQuaternion.h"
#include "MRMesh/MRVector3.h"
#include "MRPch/MRSpdlog.h"

namespace MR::SpaceMouse
{

void Controller::connect()
{
    auto& viewer = getViewerInstance();
    viewer.signals().spaceMouseMoveSignal.connect( MAKE_SLOT( &Controller::spaceMouseMove_ ) );
    viewer.signals().spaceMouseDownSignal.connect( MAKE_SLOT( &Controller::spaceMouseDown_ ) );
}

bool Controller::spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const Vector3f translateScaled = mult( translate, params_.translateScale ) * 0.02f;
    const Vector3f rotateScaled = mult( rotate, params_.rotateScale ) * 0.001f;

    const auto& viewportParams = viewport.getParameters();

    //translation
    Vector3f axisN;
    Vector3f axisX = Vector3f::plusX();
    Vector3f axisY = Vector3f::minusY();
    axisN = viewport.unprojectFromClipSpace( axisN );
    axisX = ( viewport.unprojectFromClipSpace( axisX ) - axisN );
    axisY = ( viewport.unprojectFromClipSpace( axisY ) - axisN );
    Vector3f diff( translateScaled.x * axisX + translateScaled.z * axisY );

    viewport.setCameraTranslation( viewportParams.cameraTranslation + diff * 0.1f );

    //zoom
    float  mult = pow( 0.95f, fabs( translateScaled.y ) * translateScaled.y );
    constexpr float min_angle = 0.001f;
    constexpr float max_angle = 179.99f;
    constexpr float d2r = PI_F / 180.0f;
    float angle = viewportParams.cameraViewAngle;
    angle = float( atan( tan( ( angle ) * ( d2r / 2.f ) ) * mult ) / ( d2r / 2.f ) );
    angle = std::clamp( angle, min_angle, max_angle );
    viewport.setCameraViewAngle( angle );

    //rotation
    if ( !lockRotate_ )
    {
        Quaternionf quat = (
            Quaternionf( Vector3f{ 1,0,0 }, rotateScaled.x ) *
            Quaternionf( Vector3f{ 0,0,1 }, rotateScaled.y ) *
            Quaternionf( Vector3f{ 0,-1,0 }, rotateScaled.z ) *
            viewportParams.cameraTrackballAngle
            ).normalized();
        viewport.setCameraTrackballAngle( quat );
    }

    return true;
}

bool Controller::spaceMouseDown_( int key )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    if ( showKeyDebug_ )
        spdlog::info( "SpaceMouse key down = {}", key );

    using namespace SpaceMouse;

    if ( key == int( Buttons::SMB_MENU ) )
    {
        showKeyDebug_ = !showKeyDebug_;
        return true;
    }
    else if ( key == int( Buttons::SMB_LOCK_ROT ) )
    {
        lockRotate_ = !lockRotate_;
        return true;
    }
    else if ( key == int( Buttons::SMB_FIT ) )
    {
        getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::Visible } );
        return true;
    }
    else if ( key == int( Buttons::SMB_TOP ) )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[1] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
    else if ( key == int( Buttons::SMB_RIGHT ) )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[6] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
    else if ( key == int( Buttons::SMB_FRONT ) )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[0] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
	return false;
}

}
