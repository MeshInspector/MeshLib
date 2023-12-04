#pragma once
#include "MRCustomPlugin.h"

namespace MR
{

class ObjectMesh;

class FixSpikesPlugin : public CustomPlugin
{
public:
    MRVIEWER_API FixSpikesPlugin();

    MRVIEWER_API virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    MRVIEWER_API virtual bool isAvailable( const std::vector<std::shared_ptr<const Object>>& selectedObjects ) override;

    MRVIEWER_API virtual std::string getTooltip() const override;
private:
    virtual bool onEnable_() override;

    virtual void onDisable_() override;

    std::shared_ptr<ObjectMesh> obj_;

    float minSumAngle_{0.0f};
    int maxIters_{0};
};

}
