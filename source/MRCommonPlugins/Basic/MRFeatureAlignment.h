#pragma once
#include "MRFeatureAlignmentController.h"
#include "MRFeatureAlignmentLayout.h"
#include <MRViewer/MRStatePlugin.h>

namespace MR
{
class FeatureAlignmentPlugin : public StatePlugin
{
public:
    FeatureAlignmentPlugin();
    std::string isAvailable( const std::vector<std::shared_ptr<const Object>> & ) const override;
    void drawDialog( ImGuiContext * ) override;
    FeatureAlignmentTool::Controller *controller()
    {
        return controller_.get();
    }
    const FeatureAlignmentTool::PanelDiagnostics &layout() const
    {
        return layout_;
    }

protected:
    bool onEnable_() override;
    bool onDisable_() override;

private:
    FeatureAlignmentTool::ViewerTransformHistory history_;
    std::unique_ptr<FeatureAlignmentTool::Controller> controller_;
    bool secondHole_{}, advanced_{}, diagnostics_{}, scrollToResult_{};
    std::string message_;
    bool error_{};
    FeatureAlignmentTool::PanelDiagnostics layout_;
    // Logical pixels retain the user's expanded size when UI scaling changes.
    Vector2f panelSize_{ 420, 560 };
};
} // namespace MR
