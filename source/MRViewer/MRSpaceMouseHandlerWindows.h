#pragma once
#ifdef _WIN32
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"

namespace MR
{

class SpaceMouseHandlerWindows : public SpaceMouseHandler, public MultiListener<PostFocusListener>
{
public:
    SpaceMouseHandlerWindows();

    virtual void initialize() override;
    virtual void handle() override;
    virtual void updateConnected( int jid, int event );
private:
    bool active_{ true };
    bool initialized_{ false };
    std::array<float, 6> axes_;
    std::array<unsigned char, SMB_BUTTON_COUNT> buttons_;
    int joystickIndex_{ -1 };
    const int* mapButtons_{ nullptr };
    int buttonsCount_{ 0 };

    virtual void postFocusSignal_( bool focused ) override;

    void updateConnected_();
};

}

#endif
