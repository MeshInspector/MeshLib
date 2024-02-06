#pragma once

#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRShadowsGL.h"
#include "MRViewer/MRSpaceMouseParameters.h"
#include "MRViewer/MRTouchpadParameters.h"
#include "MRMesh/MRVector4.h"
#include "MRCommonPlugins/exports.h"

namespace MR
{

class MRCOMMONPLUGINS_CLASS ViewerSettingsPlugin : public StatePlugin
{
public:

    enum class TabType
    {
        Settings,
        Viewport,
        View,
        Control,
        Count
    } activeTab_{ TabType::Settings };

    ViewerSettingsPlugin();

    virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    virtual bool blocking() const override { return false; }

    // call this function if you save/delete color theme, or change current theme outside of this plugin
    void updateThemes();

    class ComboSettings
    {
    public:
        virtual ~ComboSettings() {}
        virtual const std::string& getName() = 0;
        virtual void draw() = 0;
    };
    // move
    MRCOMMONPLUGINS_API void addComboSettings( const TabType tab, std::shared_ptr<ComboSettings> settings);

private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void drawSettingsTab_( float menuWidth, float menuScaling );
    void drawViewportTab_( float menuWidth, float menuScaling );
    void drawViewTab_( float menuWidth, float menuScaling );
    void drawControlTab_( float menuWidth, float menuScaling );

    void drawMouseSceneControlsSettings_( float menuWidth, float menuScaling );
    void drawSpaceMouseSettings_( float menuWidth, float menuScaling );
    void drawTouchpadSettings_();

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

    SpaceMouseParameters spaceMouseParams_;
#if defined(_WIN32) || defined(__APPLE__)
    bool activeMouseScrollZoom_{ false };
#endif

    TouchpadParameters touchpadParameters_;

    std::unordered_map< TabType, std::vector<std::shared_ptr<ComboSettings>>> comboSettings_;
};

}
