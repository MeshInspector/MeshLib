#pragma once
#include "MRViewerFwd.h"
#include "exports.h"
#include "MRMesh/MRSignal.h"
#include <memory>

namespace MR
{

// Interface for automatically update StatePlugins internal data
class IPluginUpdate
{
public:
    virtual ~IPluginUpdate() = default;
    // called each frame in before drawDialog
    virtual void preDrawUpdate() {}
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
class MRVIEWER_CLASS PluginCloseOnSelectedObjectRemove : public virtual IPluginUpdate
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
class MRVIEWER_CLASS PluginCloseOnChangeMesh : public virtual IPluginUpdate
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

// Helper class to update plugin if any of active object meshes or selected faces have changed
// note that events only marks plugin dirty and update happens before drawDialog function
// inherit your plugin from it
class MRVIEWER_CLASS PluginUpdateOnChangeMeshPart : public virtual IPluginUpdate
{
public:
    using UpdateFunc = std::function<void()>;
    // setup your update function that will be called if plugin is dirty in this frame
    void setUpdateFunc( UpdateFunc func ) { func_ = func; }
    MRVIEWER_API virtual void preDrawUpdate() override;
protected:
    // sets dirty initially for first update, so no need to call UpdateFunc manually
    MRVIEWER_API virtual void onPluginEnable_() override;
    // clears connections and UpdateFunc
    MRVIEWER_API virtual void onPluginDisable_() override;
private:
    bool dirty_{ false };
    UpdateFunc func_;
    std::vector<boost::signals2::scoped_connection> connections_;
};

// Helper class to close plugin if any of active object points was changed
// inherit your plugin from it
class MRVIEWER_CLASS PluginCloseOnChangePointCloud : public virtual IPluginUpdate
{
protected:
    MRVIEWER_API virtual void onPluginEnable_() override;
    MRVIEWER_API virtual void onPluginDisable_() override;
    MRVIEWER_API virtual bool shouldClose_() const override;
    // plugin can return the value to false after points change if it changed the mesh by itself and does not want to close
    bool pointCloudChanged_{ false };

private:
    std::vector<boost::signals2::scoped_connection> pointCloudChangedConnections_;
};

// Helper class to close a dialog-less plugin when the Esc key is pressed
class MRVIEWER_CLASS PluginCloseOnEscPressed : public virtual IPluginUpdate
{
protected:
    MRVIEWER_API bool shouldClose_() const override;
};

// Runs all preDrawUpdate and all shouldClose_ checks
// shouldClose_ returns true if at least on of checks was ture 
template<typename ...Updates>
class PluginUpdateOr : public Updates...
{
public:
    virtual void preDrawUpdate() override
    {
        ( Updates::preDrawUpdate(), ... );
    }
protected:
    virtual void onPluginEnable_() override
    {
        ( Updates::onPluginEnable_(), ... );
    }
    virtual void onPluginDisable_() override
    {
        // disconnect in reversed order
        [[maybe_unused]] int dummy;
        ( void )( dummy = ... = ( Updates::onPluginDisable_(), 0 ) );
    }
    virtual bool shouldClose_() const override
    {
        return ( Updates::shouldClose_() || ... );
    }
};

}