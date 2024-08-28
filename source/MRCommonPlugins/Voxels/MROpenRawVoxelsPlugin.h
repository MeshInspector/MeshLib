#pragma once
#ifndef MESHLIB_NO_VOXELS

#include "MRViewer/MRStatePlugin.h"
#include "MRVoxels/MRVoxelsLoad.h"

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

    bool autoMode_{ false };
    VoxelsLoad::RawParameters parameters_;
};
}
#endif
