#pragma once
#ifndef __EMSCRIPTEN__
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRVoxelsLoad.h"

namespace MR
{
class OpenRawVoxelsPlugin : public StatePlugin
{
public:
    OpenRawVoxelsPlugin();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

private:
    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    bool autoMode_{ true };
    VoxelsLoad::RawParameters parameters_;
};
}
#endif
