#pragma once
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRColorTheme.h"
#include "MRMesh/MRVector4.h"
#include <string>
#include <vector>

namespace MR
{

class AddCustomThemePlugin : public StatePlugin
{
public:
    AddCustomThemePlugin();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const override;
private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void updateThemeNames_();

    Json::Value makeJson_();

    void update_();
    std::string save_();

    bool applyToNewObjectsOnly_{ true };
    std::vector<Vector4f> sceneColors_;
    std::vector<Vector4f> ribbonColors_;
    std::vector<Vector4f> viewportColors_;
    // whole color theme preset
    int selectedUserPreset_{ 0 };
    std::vector<std::string> userThemesPresets_;
    // ImGui preset
    ColorTheme::Preset preset_;
    std::string themeName_;
};

}