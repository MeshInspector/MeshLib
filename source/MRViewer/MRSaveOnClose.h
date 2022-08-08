#pragma once

#include "MRViewerPlugin.h"
#include "MRViewerEventsListener.h"

namespace MR
{

// this plugin will show a message to the user if she closes the application when something is modified
class MRVIEWER_CLASS SaveOnClosePlugin : public MR::ViewerPlugin, public MultiListener<PreDrawListener,InterruptCloseListener>
{
public:
    MRVIEWER_API virtual void init( Viewer* _viewer ) override;
    MRVIEWER_API virtual void shutdown() override;

private:
    virtual void preDraw_() override;
    virtual bool interruptClose_() override;

    bool shouldClose_{false};
    bool showCloseModal_{false};
    // how long active modal will blink in seconds
    float activeModalHighlightTimer_{ 0.0f };
};

} //namespace MR
