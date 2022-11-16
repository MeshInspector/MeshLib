#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include <functional>

namespace MR
{

class SpaceMouseHandler
{
public:
    SpaceMouseHandler() = default;
    virtual ~SpaceMouseHandler() = default;

    enum Button : int
    {
        MENU,
        FIT,
        TOP,
        RIGHT,
        FRONT,
        ROLL_CW,
        CUSTOM_1,
        CUSTOM_2,
        CUSTOM_3,
        CUSTOM_4,
        ESC,
        SHIFT,
        CTRL,
        ALT,
        LOCK_ROT,
        BUTTON_COUNT
    };

    virtual void initialize() {};
    virtual void handle() {};
    virtual void updateConnected( int /*jid*/, int /*event*/ ) {};
};

}
