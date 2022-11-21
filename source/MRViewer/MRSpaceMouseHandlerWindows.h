#pragma once
#ifdef _WIN32
#include "MRSpaceMouseHandler.h"

namespace MR
{

class SpaceMouseHandlerWindows : public SpaceMouseHandler
{
public:
    virtual void initialize() override;
    virtual void handle() override;
    virtual void updateConnected( int jid, int event );
private:
    bool initialized_{ false };
    std::array<float, 6> axes_;
    std::array<unsigned char, BUTTON_COUNT> buttons_;
    int joystickIndex_{ -1 };

    virtual void updateConnected_();
};

}

#endif
