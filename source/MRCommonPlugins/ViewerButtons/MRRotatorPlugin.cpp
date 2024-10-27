#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRLine3.h"
#include <chrono>
#include <optional>

namespace MR
{

class RotatorPlugin : public StateListenerPlugin<PreDrawListener>
{
public:
    RotatorPlugin();

    void drawDialog( float menuScaling, ImGuiContext* ) override;
    bool blocking() const override { return false; }

private:
    bool onEnable_() override;
    bool onDisable_() override;
    void preDraw_() override;

    std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>> lastDrawTime_;
    float rotationSpeed_ = 5 * PI_F / 180;
};

RotatorPlugin::RotatorPlugin() :
    StateListenerPlugin( "Rotator" )
{
}

void RotatorPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 150.0f * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::SetNextItemWidth( 90.0f * menuScaling );
    UI::drag<AngleUnit>( "Speed", rotationSpeed_, 0.01f, -2 * PI_F, 2 * PI_F );
    UI::setTooltipIfHovered( "The speed of camera rotation in degrees per second. The sign of this value specifies the direction of rotation.", menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool RotatorPlugin::onEnable_()
{
    return true;
}

bool RotatorPlugin::onDisable_()
{
    lastDrawTime_.reset();
    return true;
}

void RotatorPlugin::preDraw_()
{
    auto now = std::chrono::high_resolution_clock::now();
    if ( lastDrawTime_ )
    {
        auto timePassed = std::chrono::duration<float>( *lastDrawTime_ - now ).count();
        auto & viewport = Viewport::get();
        Vector3f sceneCenter;
        if ( auto sceneBox = viewport.getSceneBox(); sceneBox.valid() )
            sceneCenter = sceneBox.center();

        viewport.cameraRotateAround(
            Line3f{ sceneCenter, viewport.getUpDirection() },
            timePassed * rotationSpeed_ );
    }
    lastDrawTime_ = now;
    incrementForceRedrawFrames();
}

MR_REGISTER_RIBBON_ITEM( RotatorPlugin )

}
