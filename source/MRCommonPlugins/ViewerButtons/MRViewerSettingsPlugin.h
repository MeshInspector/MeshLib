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

    // call this function if you save/delete color theme, or change current theme outside of this plugin
    void updateThemes();
private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void drawMouseSceneControlsSettings_( float menuScaling );

    void drawSpaceMouseSettings_( float menuScaling );

    void drawModalExitButton_( float menuScaling );

    int curSamples_{ 0 };
    int storedSamples_{ 0 };
    int maxSamples_{ 0 };
    bool needReset_{ false };
    bool gpuOverridesMSAA_{ false };

    Vector4f backgroundColor_;

    int selectedUserPreset_{ 0 };
    std::vector<std::string> userThemesPresets_;

    RibbonMenu* ribbonMenu_ = nullptr;

    Vector4f shadowColor4f_;
    std::unique_ptr<ShadowsGL> shadowGl_;

    SpaceMouseController::Params spaceMouseParams_;
#ifdef _WIN32
    bool activeMouseScrollZoom_{ false };
#endif
};

}