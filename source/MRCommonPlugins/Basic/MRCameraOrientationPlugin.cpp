#include "MRCameraOrientationPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/ImGuiHelpers.h"

namespace MR
{

CameraOrientation::CameraOrientation():
    StatePlugin( "Camera", StatePluginTabs::Basic )
{

}

void CameraOrientation::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 340 * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, &dialogIsCollapsed_, menuWidth, menuScaling ) )
        return;

    if ( viewer->viewport_list.size() > 1 )
        ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

    ImGui::DragFloatValid3( "Position", &position_.x );
    ImGui::SetTooltipIfHovered( "Location of camera focal point in world space. In case of Autofit, this location is automatically re-calculated.", menuScaling );

    ImGui::DragFloatValid3( "Direction", &direction_.x );
    ImGui::SetTooltipIfHovered( "Forward direction of the camera in world space.", menuScaling );

    ImGui::DragFloatValid3( "Up", &upDir_.x );
    ImGui::SetTooltipIfHovered( "Up direction of the camera in world space.", menuScaling );

    if ( RibbonButtonDrawer::GradientButton( "Orthonormalize", ImVec2( -1, 0 ) ) )
        upDir_ = cross( cross( direction_, upDir_ ), direction_ ).normalized();
    ImGui::SetTooltipIfHovered( "Recalculate vector to orthonormal\n"
                                "saving plane (direction, up)", menuScaling );

    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( RibbonButtonDrawer::GradientButton( "Get camera", ImVec2( ( w - p ) / 2.f, 0 ) ) )
    {
        position_ = Viewer::instanceRef().viewport().getCameraPoint();
        const auto& quaternion = Viewer::instanceRef().viewport().getParameters().cameraTrackballAngle;
        direction_ = quaternion.inverse()( Vector3f( 0.f, 0.f, -1.f ) );
        upDir_ = quaternion.inverse()( Vector3f( 0.f, 1.f, 0.f ) );
    }
    ImGui::SameLine( 0.f, p );
    if ( RibbonButtonDrawer::GradientButton( "Set camera", ImVec2( ( w - p ) / 2.f, 0 ) ) )
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
    if ( !RibbonButtonDrawer::CollapsingHeader( "Camera Presets" ) )
        return;

    const ImVec2 buttonSize = ImVec2( 60.f * scaling, 0.f );
    const float centerButtonShift = buttonSize.x + ImGui::GetStyle().ItemSpacing.x;
    const float backShift = 70.f * scaling;

    auto applyCanonicalQuaternions = [&] ( int num )
    {
        Viewer::instanceRef().viewport().setCameraTrackballAngle( getCanonicalQuaternions<float>()[num] );
        autofit_();
    };

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( RibbonButtonDrawer::GradientButton( "Top", buttonSize ) )
        applyCanonicalQuaternions( 1 );

    if ( RibbonButtonDrawer::GradientButton( "Left", buttonSize ) )
        applyCanonicalQuaternions( 4 );
    ImGui::SameLine();
    if ( RibbonButtonDrawer::GradientButton( "Front", buttonSize ) )
        applyCanonicalQuaternions( 0 );
    ImGui::SameLine();
    if ( RibbonButtonDrawer::GradientButton( "Right", buttonSize ) )
        applyCanonicalQuaternions( 6 );
    ImGui::SameLine( 0.f, backShift );
    if ( RibbonButtonDrawer::GradientButton( "Back", buttonSize ) )
        applyCanonicalQuaternions( 5 );

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + centerButtonShift );
    if ( RibbonButtonDrawer::GradientButton( "Bottom", buttonSize ) )
        applyCanonicalQuaternions( 3 );
    ImGui::SameLine();

    const float isometricShift = buttonSize.x + backShift - 20.f * scaling;
    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + isometricShift );
    if ( RibbonButtonDrawer::GradientButton( "Isometric", ImVec2( buttonSize.x + 20.f * scaling, buttonSize.y ) ) )
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
