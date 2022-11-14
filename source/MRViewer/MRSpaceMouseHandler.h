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

    struct MotionEvent
    {
        Vector3i translation;
        Vector3i rotation;
    };
    //void setMotionCallback( const std::function<void( const MotionEvent& )>& cb );

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
    enum class ButtonState
    {
        RELEASED = 0,
        PRESSED = 1,
    };
    struct ButtonEvent
    {
        int code;
        ButtonState state;
    };
    //void setButtonCallback( const std::function<void( const ButtonEvent& )>& cb );

    virtual void initialize() {};
    virtual void handle() {};

protected:
    bool initialized_{ false };
};

}
