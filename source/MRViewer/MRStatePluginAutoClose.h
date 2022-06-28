#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "exports.h"
#include <memory>

namespace MR
{

// Interface for automatically close StatePlugins
class IPluginCloseCheck
{
public:
    virtual ~IPluginCloseCheck() = default;
protected:
    // called when plugin started
    virtual void onPluginEnable_() { }
    // called when plugin stops
    virtual void onPluginDisable_() { }
    // called each frame, return true to close plugin
    virtual bool shouldClose_() const { return false; }
};

// Helper class to close plugin if any of active objects was removed from scene
// inherit your plugin from it
class MRVIEWER_CLASS PluginCloseOnSelectedObjectRemove : public virtual IPluginCloseCheck
{
protected:
    MRVIEWER_API virtual void onPluginEnable_() override;
    MRVIEWER_API virtual void onPluginDisable_() override;
    MRVIEWER_API virtual bool shouldClose_() const override;
private:
    std::vector<std::shared_ptr<MR::Object>> selectedObjs_;
};
}