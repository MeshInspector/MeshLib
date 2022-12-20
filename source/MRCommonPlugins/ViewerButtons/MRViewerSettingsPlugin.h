#pragma once
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRVector4.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRShadowsGL.h"
#include "MRViewer/MRSpaceMouseController.h"

namespace MR
{

class ViewerSettingsPlugin : public StatePlugin
{
public:
    ViewerSettingsPlugin();

    virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    virtual bool blocking() const override { return false; }

private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void drawMouseSceneControlsSettings_( float menuScaling );

    void drawDialogQuickAccessSettings_( float menuScaling );

    void drawQuickAccessList_();

    void drawSpaceMouseSettings_( float menuScaling );

    void drawModalExitButton_( float menuScaling );

    int curSamples_{ 0 };
    int storedSamples_{ 0 };
    int maxSamples_{ 0 };
    bool needReset_{ false };

    Vector4f backgroundColor_;

    int selectedUserPreset_{ 0 };
    std::vector<std::string> userThemesPresets_;

    const RibbonSchema* schema_ = nullptr;
    MenuItemsList* quickAccessList_ = nullptr;
    int maxQuickAccessSize_{ 0 };
    RibbonMenu* ribbonMenu_ = nullptr;

    Vector4f shadowColor4f_;
    std::unique_ptr<ShadowsGL> shadowGl_;

    SpaceMouseController::Params spaceMouseParams_;
    bool disableMouseScrollZoom_{ false };
};

}