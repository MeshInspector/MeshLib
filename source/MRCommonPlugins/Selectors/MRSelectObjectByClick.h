#pragma once
#include "MRViewer/MRStatePlugin.h"

namespace MR
{

class Object;

class SelectObjectByClick : public StateListenerPlugin<MouseDownListener, MouseUpListener, MouseMoveListener>
{
public:
    SelectObjectByClick();

    virtual bool blocking() const override { return false; };

    virtual void drawDialog( float, ImGuiContext* ) override;
private:
    virtual bool onMouseDown_( MouseButton button, int modifier ) override;
    virtual bool onMouseUp_( MouseButton button, int modifier ) override;
    virtual bool onMouseMove_( int x, int y ) override;

    void select_( bool up );

    bool picked_{ false };
    bool ctrl_{ false };
};

}
