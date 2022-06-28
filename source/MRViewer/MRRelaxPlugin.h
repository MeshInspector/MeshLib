#pragma once
#include "MRCustomPlugin.h"

namespace MR
{

class ObjectMesh;

class RelaxPlugin : public CustomPlugin
{
public:
    MRVIEWER_API RelaxPlugin();

    MRVIEWER_API virtual void drawDialog( float menuScaling, ImGuiContext* ctx ) override;

    MRVIEWER_API virtual bool isAvailable( const std::vector<std::shared_ptr<const Object>>& selectedObjects ) override;

    MRVIEWER_API virtual std::string getTooltip() const override;
private:
    virtual bool onEnable_() override;

    virtual void onDisable_() override;

    std::shared_ptr<ObjectMesh> obj_;
    int iters_{1};
    bool saveVolume_{true};
};

}
