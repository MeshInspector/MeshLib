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

    Vector3f translateScale_{ 100.f, 100.f, 100.f };
    Vector3f rotateScale_{ 10.f, 10.f, 10.f };
};

}
