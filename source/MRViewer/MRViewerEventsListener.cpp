#include "MRViewerEventsListener.h"
#include "MRViewer.h"

namespace MR
{

void MouseDownListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->mouseDownSignal.connect( group, MAKE_SLOT( &MouseDownListener::onMouseDown_ ), pos );
}

void MouseUpListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->mouseUpSignal.connect( group, MAKE_SLOT( &MouseUpListener::onMouseUp_ ), pos );
}

void MouseMoveListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->mouseMoveSignal.connect( group, MAKE_SLOT( &MouseMoveListener::onMouseMove_ ), pos );
}

void MouseScrollListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->mouseScrollSignal.connect( group, MAKE_SLOT( &MouseScrollListener::onMouseScroll_ ), pos );
}

void CharPressedListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->charPressedSignal.connect( group, MAKE_SLOT( &CharPressedListener::onCharPressed_ ), pos );
}

void KeyUpListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->keyUpSignal.connect( group, MAKE_SLOT( &KeyUpListener::onKeyUp_ ), pos );
}

void KeyDownListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->keyDownSignal.connect( group, MAKE_SLOT( &KeyDownListener::onKeyDown_ ), pos );
}

void KeyRepeatListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->keyRepeatSignal.connect( group, MAKE_SLOT( &KeyRepeatListener::onKeyRepeat_ ), pos );
}

void PreDrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->preDrawSignal.connect( group, MAKE_SLOT( &PreDrawListener::preDraw_ ), pos );
}

void DrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->drawSignal.connect( group, MAKE_SLOT( &DrawListener::draw_ ), pos );
}

void PostDrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->postDrawSignal.connect( group, MAKE_SLOT( &PostDrawListener::postDraw_ ), pos );
}

void SaveListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->saveSignal.connect( group, MAKE_SLOT( &SaveListener::save_ ), pos );
}

void LoadListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->loadSignal.connect( group, MAKE_SLOT( &LoadListener::load_ ), pos );
}

void DragDropListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->dragDropSignal.connect( group, MAKE_SLOT( &DragDropListener::dragDrop_ ), pos );
}

void PostResizeListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->postResizeSignal.connect( group, MAKE_SLOT( &PostResizeListener::postResize_ ), pos );
}

void InterruptCloseListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->interruptCloseSignal.connect( group, MAKE_SLOT( &InterruptCloseListener::interruptClose_ ), pos );
}

void PostRescaleListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->postRescaleSignal.connect( group, MAKE_SLOT( &PostRescaleListener::postRescale_ ), pos );
}

}