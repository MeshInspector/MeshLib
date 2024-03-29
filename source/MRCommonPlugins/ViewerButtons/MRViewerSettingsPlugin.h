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
        Quick,
        Application,
        Control,
        Viewport,
        MeasurementUnits,
        Features,
        Count
    };

    ViewerSettingsPlugin();

    virtual const std::string& uiName() const override;

    virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    virtual bool blocking() const override { return false; }

    // call this function if you save/delete color theme, or change current theme outside of this plugin
    void updateThemes();

    // basic class of external settings
    class ExternalSettings
    {
    public:
        virtual ~ExternalSettings() {}
        // returns the name of the setting, which is a unique value
        virtual const std::string& getName() = 0;
        // the function of drawing the configuration UI
        virtual void draw( float menuScaling ) = 0;
        // restore the settings to their default values
        virtual void reset() {}
    };
    // add external settings with UI combo box
    MRCOMMONPLUGINS_API void addComboSettings( const TabType tab, std::shared_ptr<ExternalSettings> settings);

private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void drawTab_( TabType tab, float menuWidth, float menuScaling );

    void drawQuickTab_( float menuWidth, float menuScaling );
    void drawApplicationTab_( float menuWidth, float menuScaling );
    void drawControlTab_( float menuWidth, float menuScaling );
    void drawViewportTab_( float menuWidth, float menuScaling );
    void drawMeasurementUnitsTab_( float menuScaling );
    void drawFeaturesTab_( float menuScaling );

    void drawThemeSelector_( float menuWidth, float menuScaling );
    void drawResetDialog_( bool activated, float menuScaling );
    void drawShadingModeCombo_( bool inGroup, float menuScaling );
    void drawProjectionModeSelector_( float menuScaling );
    void drawUpDirectionSelector_();
    void drawBackgroundButton_( bool allViewports );
    void drawRenderOptions_( float menuScaling );
    void drawShadowsOptions_( float menuWidth, float menuScaling );
    void drawMouseSceneControlsSettings_( float menuWidth, float menuScaling );
    void drawSpaceMouseSettings_( float menuWidth, float menuScaling );
    void drawTouchpadSettings_( float menuScaling );

    void drawCustomSettings_( TabType tabType, float menuScaling );

    void updateDialog_();
    void resetSettings_();

    int curSamples_{ 0 };
    int storedSamples_{ 0 };
    int maxSamples_{ 0 };
    bool needReset_{ false };
    bool gpuOverridesMSAA_{ false };

    Vector4f backgroundColor_;

    int selectedUserPreset_{ 0 };
    std::vector<std::string> userThemesPresets_;

    RibbonMenu* ribbonMenu_ = nullptr;

    std::unique_ptr<ShadowsGL> shadowGl_;

    SpaceMouseParameters spaceMouseParams_;
#if defined(_WIN32) || defined(__APPLE__)
    bool activeMouseScrollZoom_{ false };
#endif

    TouchpadParameters touchpadParameters_;

    TabType activeTab_ = TabType::Quick;

    std::array<std::vector<std::shared_ptr<ExternalSettings>>, size_t(TabType::Count)> comboSettings_;
};

}
