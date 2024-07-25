#pragma once
#include "MRViewer/MRStatePlugin.h"

namespace MR
{

class Object;

class SelectObjectByClick :
    public StateListenerPlugin<
        MouseDownListener,
        MouseUpListener,
        MouseMoveListener
    >,
    public PluginCloseOnEscPressed
{
public:
    SelectObjectByClick();

    virtual bool blocking() const override { return false; };

    virtual void drawDialog( float, ImGuiContext* ) override;
private:
    virtual bool onMouseDown_( MouseButton button, int modifiers ) override;
    virtual bool onMouseUp_( MouseButton button, int modifiers ) override;
    virtual bool onMouseMove_( int x, int y ) override;

    void select_( bool up );

    bool picked_{ false };
    bool ctrl_{ false };
};

}
