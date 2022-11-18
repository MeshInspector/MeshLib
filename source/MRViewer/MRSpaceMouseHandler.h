#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include <functional>

namespace MR
{

/// enumeration all spacemouse buttons
enum SpaceMouseButtons : int
{
    SMB_MENU,

    SMB_ESC,
    SMB_ENTER,
    SMB_TAB,
    SMB_SHIFT,
    SMB_CTRL,
    SMB_ALT,
    SMB_SPACE,
    SMB_DELETE,

    SMB_CUSTOM_1,
    SMB_CUSTOM_2,
    SMB_CUSTOM_3,
    SMB_CUSTOM_4,
    SMB_CUSTOM_5,
    SMB_CUSTOM_6,
    SMB_CUSTOM_7,
    SMB_CUSTOM_8,
    SMB_CUSTOM_9,
    SMB_CUSTOM_10,
    SMB_CUSTOM_11,
    SMB_CUSTOM_12,

    SMB_FIT,
    SMB_TOP,
    SMB_RIGHT,
    SMB_FRONT,
    SMB_ROLL_CW, // roll clockwise
    SMB_LOCK_ROT,

    SMB_BTN_V1,
    SMB_BTN_V2,
    SMB_BTN_V3,
    SMB_ISO1,

    SMB_BUTTON_COUNT
};


/// class to handle spacemouse
class SpaceMouseHandler
{
public:
    SpaceMouseHandler() = default;
    virtual ~SpaceMouseHandler() = default;

    /// initialize device
    virtual void initialize() {};

    /// handle device state and call Viewer signals
    virtual void handle() {};

    /// update after connect / disconnect devices
    virtual void updateConnected( int /*jid*/, int /*event*/ ) {};
};

}
