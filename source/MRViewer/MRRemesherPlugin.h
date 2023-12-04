#pragma once
#include "MRCustomPlugin.h"

namespace MR
{

class ObjectMesh;

class RemesherPlugin : public CustomPlugin
{
public:
    MRVIEWER_API RemesherPlugin();

    MRVIEWER_API virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    MRVIEWER_API virtual bool isAvailable( const std::vector<std::shared_ptr<const Object>>& selectedObjects ) override;

    MRVIEWER_API virtual std::string getTooltip() const override;
private:
    virtual bool onEnable_() override;

    virtual void onDisable_() override;

    float maxDeviation_{1000};
    int iters_{100};

    std::shared_ptr<ObjectMesh> obj_;
};

}
