#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"

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

    float rotationSpeed_ = 5 * PI_F / 180;
    bool rotateCamera_ = true;
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

    UI::checkbox( "Rotate Camera", &rotateCamera_ ); 
    UI::setTooltipIfHovered( "If selected then camera is rotated around scene's center. Otherwise selected objects are rotated, each around its center.", menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool RotatorPlugin::onEnable_()
{
    return true;
}

bool RotatorPlugin::onDisable_()
{
    return true;
}

void RotatorPlugin::preDraw_()
{
    auto & viewport = Viewport::get();
    Vector3f sceneCenter;
    if ( auto sceneBox = viewport.getSceneBox(); sceneBox.valid() )
        sceneCenter = sceneBox.center();

    const auto deltaAngle = ImGui::GetIO().DeltaTime * rotationSpeed_;

    if ( rotateCamera_ )
    {
        viewport.cameraRotateAround( Line3f{ sceneCenter, viewport.getUpDirection() }, -deltaAngle );
    }
    else
    {
        const auto rotMat = Matrix3f::rotation( viewport.getUpDirection(), deltaAngle );
        for ( const auto & obj : getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            const auto wXf = obj->worldXf();
            const auto wCenter = obj->getWorldTreeBox().center();
            obj->setWorldXf( AffineXf3f::xfAround( rotMat, wCenter ) * wXf );
        }
    }

    incrementForceRedrawFrames();
}

MR_REGISTER_RIBBON_ITEM( RotatorPlugin )

}
