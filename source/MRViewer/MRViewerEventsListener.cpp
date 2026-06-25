#include "MRViewerEventsListener.h"
#include "MRViewer.h"
#include "MRViewerSignals.h"
#include "MRMakeSlot.h"

namespace MR
{

void MouseDownListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().mouseDownSignal.connect( group, MAKE_SLOT( &MouseDownListener::onMouseDown_ ), pos );
}

void MouseUpListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().mouseUpSignal.connect( group, MAKE_SLOT( &MouseUpListener::onMouseUp_ ), pos );
}

void MouseMoveListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().mouseMoveSignal.connect( group, MAKE_SLOT( &MouseMoveListener::onMouseMove_ ), pos );
}

void MouseScrollListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().mouseScrollSignal.connect( group, MAKE_SLOT( &MouseScrollListener::onMouseScroll_ ), pos );
}

void MouseClickListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().mouseClickSignal.connect( group, MAKE_SLOT( &MouseClickListener::onMouseClick_ ), pos );
}

void DragStartListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragStartSignal.connect( group, MAKE_SLOT( &DragStartListener::onDragStart_ ), pos );
}

void DragEndListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragEndSignal.connect( group, MAKE_SLOT( &DragEndListener::onDragEnd_ ), pos );
}

void DragListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragSignal.connect( group, MAKE_SLOT( &DragListener::onDrag_ ), pos );
}

void CharPressedListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().charPressedSignal.connect( group, MAKE_SLOT( &CharPressedListener::onCharPressed_ ), pos );
}

void KeyUpListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().keyUpSignal.connect( group, MAKE_SLOT( &KeyUpListener::onKeyUp_ ), pos );
}

void KeyDownListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().keyDownSignal.connect( group, MAKE_SLOT( &KeyDownListener::onKeyDown_ ), pos );
}

void KeyRepeatListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().keyRepeatSignal.connect( group, MAKE_SLOT( &KeyRepeatListener::onKeyRepeat_ ), pos );
}

void PreSetupViewListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().preSetupViewSignal.connect( group, MAKE_SLOT( &PreSetupViewListener::preSetupView_ ), pos );
}

void PreDrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().preDrawSignal.connect( group, MAKE_SLOT( &PreDrawListener::preDraw_ ), pos );
}

void DrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().drawSignal.connect( group, MAKE_SLOT( &DrawListener::draw_ ), pos );
}

void PostDrawListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().postDrawSignal.connect( group, MAKE_SLOT( &PostDrawListener::postDraw_ ), pos );
}

void DragDropListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragDropSignal.connect( group, MAKE_SLOT( &DragDropListener::dragDrop_ ), pos );
}

void DragEntranceListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragEntranceSignal.connect( group, MAKE_SLOT( &DragEntranceListener::dragEntrance_ ), pos );
}


void DragOverListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().dragOverSignal.connect( group, MAKE_SLOT( &DragOverListener::dragOver_ ), pos );
}

void PostResizeListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().postResizeSignal.connect( group, MAKE_SLOT( &PostResizeListener::postResize_ ), pos );
}

void InterruptCloseListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().interruptCloseSignal.connect( group, MAKE_SLOT( &InterruptCloseListener::interruptClose_ ), pos );
}

void PostRescaleListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().postRescaleSignal.connect( group, MAKE_SLOT( &PostRescaleListener::postRescale_ ), pos );
}

void TouchStartListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchStartSignal.connect( group, MAKE_SLOT( &TouchStartListener::onTouchStart_ ), pos );
}

void TouchMoveListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchMoveSignal.connect( group, MAKE_SLOT( &TouchMoveListener::onTouchMove_ ), pos );
}

void TouchEndListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchEndSignal.connect( group, MAKE_SLOT( &TouchEndListener::onTouchEnd_ ), pos );
}

void SpaceMouseMoveListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().spaceMouseMoveSignal.connect( group, MAKE_SLOT( &SpaceMouseMoveListener::spaceMouseMove_ ), pos );
}

void SpaceMouseDownListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().spaceMouseDownSignal.connect( group, MAKE_SLOT( &SpaceMouseDownListener::spaceMouseDown_ ), pos );
}

void SpaceMouseUpListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().spaceMouseUpSignal.connect( group, MAKE_SLOT( &SpaceMouseUpListener::spaceMouseUp_ ), pos );
}

void TouchpadRotateGestureBeginListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadRotateGestureBeginSignal.connect( group, MAKE_SLOT( &TouchpadRotateGestureBeginListener::touchpadRotateGestureBegin_ ), pos );
}

void TouchpadRotateGestureUpdateListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadRotateGestureUpdateSignal.connect( group, MAKE_SLOT( &TouchpadRotateGestureUpdateListener::touchpadRotateGestureUpdate_ ), pos );
}

void TouchpadRotateGestureEndListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadRotateGestureEndSignal.connect( group, MAKE_SLOT( &TouchpadRotateGestureEndListener::touchpadRotateGestureEnd_ ), pos );
}

void TouchpadSwipeGestureBeginListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadSwipeGestureBeginSignal.connect( group, MAKE_SLOT( &TouchpadSwipeGestureBeginListener::touchpadSwipeGestureBegin_ ), pos );
}

void TouchpadSwipeGestureUpdateListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadSwipeGestureUpdateSignal.connect( group, MAKE_SLOT( &TouchpadSwipeGestureUpdateListener::touchpadSwipeGestureUpdate_ ), pos );
}

void TouchpadSwipeGestureEndListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadSwipeGestureEndSignal.connect( group, MAKE_SLOT( &TouchpadSwipeGestureEndListener::touchpadSwipeGestureEnd_ ), pos );
}

void TouchpadZoomGestureBeginListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadZoomGestureBeginSignal.connect( group, MAKE_SLOT( &TouchpadZoomGestureBeginListener::touchpadZoomGestureBegin_ ), pos );
}

void TouchpadZoomGestureUpdateListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadZoomGestureUpdateSignal.connect( group, MAKE_SLOT( &TouchpadZoomGestureUpdateListener::touchpadZoomGestureUpdate_ ), pos );
}

void TouchpadZoomGestureEndListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().touchpadZoomGestureEndSignal.connect( group, MAKE_SLOT( &TouchpadZoomGestureEndListener::touchpadZoomGestureEnd_ ), pos );
}

void PostFocusListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().postFocusSignal.connect( group, MAKE_SLOT( &PostFocusListener::postFocus_ ), pos );
}

void CursorEntranceListener::connect( Viewer* viewer, int group, boost::signals2::connect_position pos )
{
    if ( !viewer )
        return;
    connection_ = viewer->signals().cursorEntranceSignal.connect( group, MAKE_SLOT( &CursorEntranceListener::cursorEntrance_ ), pos );
}

}
