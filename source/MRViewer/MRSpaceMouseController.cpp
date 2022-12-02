#include "MRSpaceMouseController.h"
#include "MRViewer.h"
#include "MRViewerInstance.h"
#include "MRViewport.h"
#include "MRSpaceMouseHandler.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRQuaternion.h"
#include "MRMesh/MRVector3.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

void SpaceMouseController::connect()
{
    auto& viewer = getViewerInstance();
    viewer.spaceMouseMoveSignal.connect( MAKE_SLOT( &SpaceMouseController::spaceMouseMove_ ) );
    viewer.spaceMouseDownSignal.connect( MAKE_SLOT( &SpaceMouseController::spaceMouseDown_ ) );
}

void SpaceMouseController::setParams( const Params& newParams )
{
    params = newParams;
    for ( int i = 0; i < 3; ++i )
    {
        float sign = params.translateScale[i] < 0 ? -1.f : 1.f;
        if ( params.translateScale[i] * sign < 50 )
            params.translateScale[i] = ( 25.f + params.translateScale[i] * sign / 2.f ) * sign;
        sign = params.rotateScale[i] < 0 ? -1.f : 1.f;
        if ( params.rotateScale[i] < 50 )
            params.rotateScale[i] = ( 25.f + params.rotateScale[i] * sign / 2.f ) * sign;
    }
}

SpaceMouseController::Params SpaceMouseController::getParams() const
{
    Params out = params;
    for ( int i = 0; i < 3; ++i )
    {
        float sign = out.translateScale[i] < 0 ? -1.f : 1.f;
        if ( out.translateScale[i] * sign < 50 )
            out.translateScale[i] = ( out.translateScale[i] * sign - 25.f ) * 2.f * sign;
        sign = out.rotateScale[i] < 0 ? -1.f : 1.f;
        if ( out.rotateScale[i] * sign < 50 )
            out.rotateScale[i] = ( out.rotateScale[i] * sign - 25.f ) * 2.f * sign;
    }
    return out;
}

bool SpaceMouseController::spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    const Vector3f translateScaled = mult( translate, params.translateScale ) * 0.02f;
    const Vector3f rotateScaled = mult( rotate, params.rotateScale ) * 0.001f;

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

bool SpaceMouseController::spaceMouseDown_( int key )
{
    auto& viewer = getViewerInstance();
    auto& viewport = viewer.viewport();

    if ( showKeyDebug_ )
        spdlog::info( "SpaceMouse key down = {}", key );

    if ( key == SMB_MENU )
    {
        showKeyDebug_ = !showKeyDebug_;
        return true;
    }
    else if ( key == SMB_LOCK_ROT )
    {
        lockRotate_ = !lockRotate_;
        return true;
    }
    else if ( key == SMB_FIT )
    {
        getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::Visible } );
        return true;
    }
    else if ( key == SMB_TOP )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[1] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
    else if ( key == SMB_RIGHT )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[6] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
    else if ( key == SMB_FRONT )
    {
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[0] );
        viewport.preciseFitDataToScreenBorder( { 0.9f } );
        return true;
    }
	return false;
}

}
