#include "MRCameraOrientationPlugin.h"
#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRRibbonButtonDrawer.h"
#include "MRViewer/MRI18n.h"

namespace MR
{

CameraOrientation::CameraOrientation():
    StatePlugin( "Camera", StatePluginTabs::Test )
{

}

void CameraOrientation::drawDialog( ImGuiContext* )
{
    auto menuWidth = 340 * UI::scale();
    if ( !ImGuiBeginWindow_( { .width = menuWidth } ) )
        return;

    if ( viewer->viewport_list.size() > 1 )
        ImGui::Text( "%s: %d", _tr( "Current viewport" ), viewer->viewport().id.value() );

    UI::drag<LengthUnit>( _tr( "Position" ), position_ );
    UI::setTooltipIfHovered( _tr( "Location of camera focal point in world space. In case of Autofit, this location is automatically re-calculated." ) );

    UI::drag<NoUnit>( _tr( "Direction" ), direction_ );
    UI::setTooltipIfHovered( _tr( "Forward direction of the camera in world space." ) );

    UI::drag<NoUnit>( _tr( "Up" ), upDir_ );
    UI::setTooltipIfHovered( _tr( "Up direction of the camera in world space." ) );

    if ( UI::button( _tr( "Orthonormalize" ), Vector2f( -1, 0 ) ) )
        upDir_ = cross( cross( direction_, upDir_ ), direction_ ).normalized();
    UI::setTooltipIfHovered( _tr( "Recalculate vector to orthonormal\n"
                                "saving plane (direction, up)" ) );

    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( UI::button( _tr( "Get camera" ), Vector2f( ( w - p ) / 2.f, 0 ) ) )
        getCamera_();
    ImGui::SameLine( 0.f, p );
    if ( UI::button( _tr( "Set camera" ), Vector2f( ( w - p ) / 2.f, 0 ) ) )
    {
        if ( cross( direction_, upDir_ ).length() > std::numeric_limits<float>::min() * 10.f )
        {
            upDir_ = cross( cross( direction_, upDir_ ), direction_ ).normalized();
            Viewer::instanceRef().viewport().setCameraPoint( position_ );
            Viewer::instanceRef().viewport().cameraLookAlong( direction_, upDir_ );
            autofit_();
        }
    }

    if ( UI::checkbox( _tr( "Autofit" ), &isAutofit_ ) )
        autofit_();
    UI::setTooltipIfHovered( _tr( "If enabled, it automatically selects best camera location to see whole scene in the viewport." ) );

    ImGui::PushItemWidth( 80 * UI::scale() );
    auto params = viewer->viewport().getParameters();
    auto fov = params.cameraViewAngle;
    UI::drag<AngleUnit>( _tr( "Camera FOV" ), fov, 0.1f, 0.01f, 179.99f, { .sourceUnit = AngleUnit::degrees } ); // `fov` is stored in degrees?!
    viewer->viewport().setCameraViewAngle( fov );

    // Orthographic view
    bool orth = params.orthographic;
    UI::checkbox( _tr( "Orthographic view" ), &orth );
    viewer->viewport().setOrthographic( orth );
    ImGui::PopItemWidth();

    drawCameraPresets_();

    ImGui::EndCustomStatePlugin();
}

void CameraOrientation::getCamera_()
{
    auto & viewport = Viewport::get();
    position_ = viewport.getCameraPoint();
    direction_ = -viewport.getBackwardDirection();
    upDir_ = viewport.getUpDirection();
}

bool CameraOrientation::onEnable_()
{
    getCamera_();
    return true;
}

void CameraOrientation::drawCameraPresets_()
{
    if ( !RibbonButtonDrawer::CustomCollapsingHeader( _tr( "Camera Presets" ) ) )
        return;

    const Vector2f buttonSize( 60.f * UI::scale(), 0.f );
    const float centerButtonShift = buttonSize.x + ImGui::GetStyle().ItemSpacing.x;
    auto width = ImGui::GetContentRegionAvail().x + ImGui::GetStyle().WindowPadding.x;
    const float backPosition = width - buttonSize.x;

    auto applyQuaternion = [&] ( const Quaternionf & q )
    {
        Viewer::instanceRef().viewport().setCameraTrackballAngle( q );
        autofit_();
    };

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( UI::button( _tr( "Top" ), buttonSize ) )
        applyQuaternion( Quaternionf() );

    if ( UI::button( _tr( "Left" ), buttonSize ) )
        applyQuaternion( Quaternionf( Vector3f(-1, 1, 1 ), 2 * PI_F / 3 ) );
    ImGui::SameLine();
    if ( UI::button( _tr( "Front" ), buttonSize ) )
        applyQuaternion( Quaternionf( Vector3f::plusX(),  -PI2_F ) );
    ImGui::SameLine();
    if ( UI::button( _tr( "Right" ), buttonSize ) )
        applyQuaternion( Quaternionf( Vector3f(-1,-1,-1 ), 2 * PI_F / 3 ) );
    ImGui::SameLine( backPosition );
    if ( UI::button( _tr( "Back" ), buttonSize ) )
        applyQuaternion( Quaternionf( Vector3f( 0, 1, 1 ), PI_F ) );

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( UI::button( _tr( "Bottom" ), buttonSize ) )
        applyQuaternion( Quaternionf( Vector3f::plusX(),   PI_F ) );

    const float isometricPos = width - ( buttonSize.x + 20.f * UI::scale() );
    ImGui::SameLine( isometricPos );
    if ( UI::button( _tr( "Isometric" ), Vector2f( buttonSize.x + 20.f * UI::scale(), buttonSize.y ) ) )
    {
        Viewer::instanceRef().viewport().cameraLookAlong( Vector3f( -1.f, -1.f, -1.f ), Vector3f( -1, -1, 2 ) );
        autofit_();
    }
}

void CameraOrientation::autofit_()
{
    if ( isAutofit_ )
    {
        Viewer::instanceRef().viewport().fitData( 1.f, false );
        position_ = Viewer::instanceRef().viewport().getCameraPoint();
    }
}

MR_REGISTER_RIBBON_ITEM( CameraOrientation )

}
