#pragma once
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRMoveObjectByMouseImpl.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRAffineXf3.h"
#include "imgui.h"

namespace MR
{

class Object;

class MoveObjectByMouse : public StateListenerPlugin<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    MoveObjectByMouse();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;

    virtual bool blocking() const override { return false; };

private:
    virtual bool onMouseDown_( MouseButton btn, int modifiers ) override;
    virtual bool onMouseMove_( int x, int y ) override;
    virtual bool onMouseUp_( MouseButton btn, int modifiers ) override;

    MoveObjectByMouseImpl moveByMouse_;
};

}
