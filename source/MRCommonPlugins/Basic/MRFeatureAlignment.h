#pragma once
#include "MRFeatureAlignmentController.h"
#include <MRMesh/MRVector2.h>
#include <MRViewer/MRStatePlugin.h>

namespace MR
{
class FeatureAlignmentPlugin : public StatePlugin
{
public:
    FeatureAlignmentPlugin();
    std::string isAvailable( const std::vector<std::shared_ptr<const Object>> & ) const override;
    void updateSelection( const std::vector<std::shared_ptr<const Object>> & ) override;
    void drawDialog( ImGuiContext * ) override;

protected:
    bool onEnable_() override;
    bool onDisable_() override;
    bool shouldClose_() const override;

private:
    std::unique_ptr<FeatureAlignmentTool::Controller> controller_;
    FeatureAlignment::Settings settings_;
    bool secondHole_{}, advanced_{}, diagnostics_{}, scrollToResult_{};
    std::string message_;
    bool error_{};
    // Logical pixels retain the user's expanded size when UI scaling changes.
    Vector2f panelSize_{ 420, 240 };
};
} // namespace MR
