#pragma once

#include "MRMesh/MRMeshFwd.h"
#include <functional>
#include <string>

namespace MR
{

/// enumeration all spacemouse buttons
enum SpaceMouseButtons : int
{
    SMB_NO = -1,
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


/// base class for handler of spacemouse devices
class SpaceMouseHandler
{
public:
    virtual ~SpaceMouseHandler() = default;

    /// initialize device
    /// \param deviceSignal every device-related event will be sent here: find, connect, disconnect
    [[nodiscard]] virtual bool initialize( std::function<void(const std::string&)> deviceSignal = {} ) = 0;

    /// handle device state and call Viewer signals
    virtual void handle() = 0;
};

} //namespace MR
