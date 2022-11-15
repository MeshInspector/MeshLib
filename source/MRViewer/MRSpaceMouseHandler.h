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
        MENU = 0,
        FIT = 1,
        TOP = 2,
        RIGHT = 4,
        FRONT = 5,
        ROLL_CW = 8,
        CUSTOM_1 = 12,
        CUSTOM_2 = 13,
        CUSTOM_3 = 14,
        CUSTOM_4 = 15,
        ESC = 22,
        SHIFT = 24,
        CTRL = 25,
        ALT = 23,
        LOCK_ROT = 26,
    };

    virtual void initialize() {};
    virtual void handle() {};
};

}
