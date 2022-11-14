#pragma once
#include "MRSpaceMouseHandler.h"

namespace MR
{

class SpaceMouseHandlerWindows : public SpaceMouseHandler
{
public:
    SpaceMouseHandlerWindows();

    virtual void initialize() override;
    virtual void handle() override;
private:
    std::array<float, 6> axes_;

    std::array<float, 6> axesScale_;
};

}
