#pragma once
#include "MRMeshViewerPlugin.h"
#include "MRMeshViewer.h"
#include "exports.h"
#include "MRViewerEventsListener.h"
#include <memory>

namespace MR
{

class ObjectMesh;

class MRVIEWER_CLASS DemoPlugin : public MR::ViewerPlugin, public MultiListener<PreDrawListener,DrawListener,InterruptCloseListener>
{
public:
    virtual void MRVIEWER_API init( Viewer* _viewer ) override;
    virtual void MRVIEWER_API shutdown() override;

private:
    virtual void draw_() override;
    virtual void preDraw_() override;
    virtual bool interruptClose_() override;

    std::unique_ptr<ObjectMesh> demoSphere_;

    bool shouldClose_{false};
    bool showCloseModal_{false};

};

MRVIEWER_API extern DemoPlugin DemoPluginInstance;

} //namespace MR
