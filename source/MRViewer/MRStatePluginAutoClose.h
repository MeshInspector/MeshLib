#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "exports.h"
#include <boost/signals2/signal.hpp>
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

// Helper class to close plugin if any of active object meshes was changed
// inherit your plugin from it
class MRVIEWER_CLASS PluginCloseOnChangeMesh : public virtual IPluginCloseCheck
{
protected:
    MRVIEWER_API virtual void onPluginEnable_() override;
    MRVIEWER_API virtual void onPluginDisable_() override;
    MRVIEWER_API virtual bool shouldClose_() const override;
    // plugin can return the value to false after mesh change if it changed the mesh by itself and does not want to close
    bool meshChanged_{ false };

private:
    std::vector<boost::signals2::scoped_connection> meshChangedConnections_;
};

// checks that at least one of argument checks is true
template<typename ...Checks>
class PluginCloseOrCheck : virtual public Checks...
{
protected:
    virtual void onPluginEnable_() override
    {
        ( Checks::onPluginEnable_(), ... );
    }
    virtual void onPluginDisable_() override
    {
        ( ..., Checks::onPluginDisable_() );
    }
    virtual bool shouldClose_() const override
    {
        return ( Checks::shouldClose_() || ... );
    }
};

}