#pragma once

#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRShadowsGL.h"
#include "MRViewer/MRSpaceMouseParameters.h"
#include "MRViewer/MRTouchpadParameters.h"
#include "MRMesh/MRVector4.h"
#include "MRCommonPlugins/exports.h"
#include "MRMruFormatParameters.h"

namespace MR
{

class MRVIEWER_CLASS ViewerSettingsPlugin : public StatePlugin
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

    virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    virtual bool blocking() const override { return false; }

    // call this function if you save/delete color theme, or change current theme outside of this plugin
    MRVIEWER_API void updateThemes();

    // basic class of external settings
    class ExternalSettings
    {
    public:
        virtual ~ExternalSettings() {}
        // returns the name of the setting, which is a unique value
        virtual const std::string& getName() const = 0;
        // the function of drawing the configuration UI
        virtual void draw( float menuScaling ) = 0;
        // restore the settings to their default values
        virtual void reset() {}
        // if not overriden this setting will be drawn in tools block
        virtual const char* separatorName() const { return "Tools"; }
    };

    /// add external settings with UI combo box
    MRVIEWER_API void addComboSettings( const TabType tab, std::shared_ptr<ExternalSettings> settings );

    /// delete external settings with UI combo box
    MRVIEWER_API void delComboSettings( const TabType tab, const ExternalSettings * settings );

    /// returns instance of this plugin if it is registered
    /// nullptr otherwise
    MRVIEWER_API static ViewerSettingsPlugin* instance();

    /// changes active tab
    MRVIEWER_API void setActiveTab( TabType tab );
private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void drawTab_( float menuWidth, float menuScaling );

    void drawQuickTab_( float menuWidth, float menuScaling );
    void drawApplicationTab_( float menuWidth, float menuScaling );
    void drawControlTab_( float menuWidth, float menuScaling );
    void drawViewportTab_( float menuWidth, float menuScaling );
    void drawMeasurementUnitsTab_( float menuScaling );
    void drawFeaturesTab_( float menuScaling );

    void drawThemeSelector_( float menuScaling );
    void drawResetDialog_( bool activated, float menuScaling );
    void drawShadingModeCombo_( bool inGroup, float menuScaling, float toolWidth );
    void drawProjectionModeSelector_( float menuScaling, float toolWidth );
    void drawUpDirectionSelector_();
    void drawBackgroundButton_( bool allViewports );
    void drawRenderOptions_( float menuScaling );
    void drawShadowsOptions_( float menuWidth, float menuScaling );
    void drawMouseSceneControlsSettings_( float menuWidth, float menuScaling );
    void drawSpaceMouseSettings_( float menuWidth, float menuScaling );
    void drawTouchpadSettings_( float menuScaling );

    void drawMruInnerFormats_( float menuWidth, float menuScaling );

    void drawGlobalSettings_( float buttonWidth, float menuScaling );
    void drawCustomSettings_( const std::string& separatorName, bool needSeparator, float menuScaling );
    void drawSeparator_( const std::string& separatorName, float menuScaling );


    void updateDialog_();
    void resetSettings_();

    int storedSamples_{ 0 };
    int maxSamples_{ 0 };
    bool gpuOverridesMSAA_{ false };
    float tempUserScaling_{ 1.0f };

    Vector4f backgroundColor_;

    int selectedUserPreset_{ 0 };
    std::vector<std::string> userThemesPresets_;

    std::unique_ptr<ShadowsGL> shadowGl_;

    SpaceMouseParameters spaceMouseParams_;
#if defined(_WIN32) || defined(__APPLE__)
    bool activeMouseScrollZoom_{ false };
#endif

    TouchpadParameters touchpadParameters_;

    TabType activeTab_ = TabType::Quick;
    TabType orderedTab_ = TabType::Count; // invalid

    std::array<std::vector<std::shared_ptr<ExternalSettings>>, size_t(TabType::Count)> comboSettings_;

    MruFormatParameters mruFormatParameters_;
};

}
