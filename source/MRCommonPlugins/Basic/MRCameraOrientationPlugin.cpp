#include "MRCameraOrientationPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"

namespace MR
{

CameraOrientation::CameraOrientation():
    StatePlugin( "Camera", StatePluginTabs::Basic )
{

}

void CameraOrientation::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 340 * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    if ( viewer->viewport_list.size() > 1 )
        ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

    ImGui::DragFloatValid3( "Position", &position_.x );
    ImGui::SetTooltipIfHovered( "Location of camera focal point in world space. In case of Autofit, this location is automatically re-calculated.", menuScaling );

    ImGui::DragFloatValid3( "Direction", &direction_.x );
    ImGui::SetTooltipIfHovered( "Forward direction of the camera in world space.", menuScaling );

    ImGui::DragFloatValid3( "Up", &upDir_.x );
    ImGui::SetTooltipIfHovered( "Up direction of the camera in world space.", menuScaling );

    if ( UI::button( "Orthonormalize", Vector2f( -1, 0 ) ) )
        upDir_ = cross( cross( direction_, upDir_ ), direction_ ).normalized();
    ImGui::SetTooltipIfHovered( "Recalculate vector to orthonormal\n"
                                "saving plane (direction, up)", menuScaling );

    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( UI::button( "Get camera", Vector2f( ( w - p ) / 2.f, 0 ) ) )
    {
        position_ = Viewer::instanceRef().viewport().getCameraPoint();
        const auto& quaternion = Viewer::instanceRef().viewport().getParameters().cameraTrackballAngle;
        direction_ = quaternion.inverse()( Vector3f( 0.f, 0.f, -1.f ) );
        upDir_ = quaternion.inverse()( Vector3f( 0.f, 1.f, 0.f ) );
    }
    ImGui::SameLine( 0.f, p );
    if ( UI::button( "Set camera", Vector2f( ( w - p ) / 2.f, 0 ) ) )
    {
        if ( cross( direction_, upDir_ ).length() > std::numeric_limits<float>::min() * 10.f )
        {
            upDir_ = cross( cross( direction_, upDir_ ), direction_ ).normalized();
            Viewer::instanceRef().viewport().setCameraPoint( position_ );
            Viewer::instanceRef().viewport().cameraLookAlong( direction_, upDir_ );
            autofit_();
        }
    }

    if ( RibbonButtonDrawer::GradientCheckbox( "Autofit", &isAutofit_ ) )
        autofit_();
    ImGui::SetTooltipIfHovered( "If enabled, it automatically selects best camera location to see whole scene in the viewport.", menuScaling );

    ImGui::PushItemWidth( 80 * menuScaling );
    auto params = viewer->viewport().getParameters();
    auto fov = params.cameraViewAngle;
    ImGui::DragFloatValid( "Camera FOV", &fov, 0.001f, 0.01f, 179.99f );
    viewer->viewport().setCameraViewAngle( fov );

    // Orthographic view
    bool orth = params.orthographic;
    RibbonButtonDrawer::GradientCheckbox( "Orthographic view", &orth );
    viewer->viewport().setOrthographic( orth );
    ImGui::PopItemWidth();

    drawCameraPresets_( menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool CameraOrientation::onEnable_()
{
    position_ = Viewer::instanceRef().viewport().getCameraPoint();
    const auto& quaternion = Viewer::instanceRef().viewport().getParameters().cameraTrackballAngle;
    direction_ = quaternion.inverse()( Vector3f( 0.f, 0.f, -1.f ) );
    upDir_ = quaternion.inverse()( Vector3f( 0.f, 1.f, 0.f ) );
    return true;
}

void CameraOrientation::drawCameraPresets_( float scaling )
{
    if ( !RibbonButtonDrawer::CustomCollapsingHeader( "Camera Presets" ) )
        return;

    const Vector2f buttonSize( 60.f * scaling, 0.f );
    const float centerButtonShift = buttonSize.x + ImGui::GetStyle().ItemSpacing.x;
    auto width = ImGui::GetContentRegionAvail().x + ImGui::GetStyle().WindowPadding.x;
    const float backPosition = width - buttonSize.x;

    auto applyCanonicalQuaternions = [&] ( int num )
    {
        Viewer::instanceRef().viewport().setCameraTrackballAngle( getCanonicalQuaternions<float>()[num] );
        autofit_();
    };

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( UI::button( "Top", buttonSize ) )
        applyCanonicalQuaternions( 1 );

    if ( UI::button( "Left", buttonSize ) )
        applyCanonicalQuaternions( 4 );
    ImGui::SameLine();
    if ( UI::button( "Front", buttonSize ) )
        applyCanonicalQuaternions( 0 );
    ImGui::SameLine();
    if ( UI::button( "Right", buttonSize ) )
        applyCanonicalQuaternions( 6 );
    ImGui::SameLine( backPosition );
    if ( UI::button( "Back", buttonSize ) )
        applyCanonicalQuaternions( 5 );

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( UI::button( "Bottom", buttonSize ) )
        applyCanonicalQuaternions( 3 );

    const float isometricPos = width - ( buttonSize.x + 20.f * scaling );
    ImGui::SameLine( isometricPos );
    if ( UI::button( "Isometric", Vector2f( buttonSize.x + 20.f * scaling, buttonSize.y ) ) )
    {
        Viewer::instanceRef().viewport().cameraLookAlong( Vector3f( -1.f, -1.f, -1.f ), Vector3f( -1.f, 2.f, -1.f ) );
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
