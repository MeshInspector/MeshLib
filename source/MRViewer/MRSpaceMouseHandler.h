#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include <functional>

namespace MR
{

/// class to handle spacemouse
class SpaceMouseHandler
{
public:
    SpaceMouseHandler() = default;
    virtual ~SpaceMouseHandler() = default;

    enum Button : int
    {
        MENU,

        ESC,
        ENTER,
        TAB,
        SHIFT,
        CTRL,
        ALT,
        SPACE,
        //DELETE,

        CUSTOM_1,
        CUSTOM_2,
        CUSTOM_3,
        CUSTOM_4,
        CUSTOM_5,
        CUSTOM_6,
        CUSTOM_7,
        CUSTOM_8,
        CUSTOM_9,
        CUSTOM_10,
        CUSTOM_11,
        CUSTOM_12,

        FIT,
        TOP,
        RIGHT,
        FRONT,
        ROLL_CW, // roll clockwise
        LOCK_ROT,

        BTN_V1,
        BTN_V2,
        BTN_V3,
        ISO1,

        BUTTON_COUNT
    };

    /// initialize device
    virtual void initialize() {};

    /// handle device state and call Viewer signals
    virtual void handle() {};

    /// update after connect / disconnect devices
    virtual void updateConnected( int /*jid*/, int /*event*/ ) {};
};

}
