#pragma once

#include "MRSignalCombiners.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

struct ViewerSignals
{
    using SignalStopHandler = StopOnTrueCombiner;
    // Mouse events
    using MouseUpDownSignal = boost::signals2::signal<bool( MouseButton btn, int modifier ), SignalStopHandler>;
    using MouseMoveSignal = boost::signals2::signal<bool( int x, int y ), SignalStopHandler>;
    using MouseScrollSignal = boost::signals2::signal<bool( float delta ), SignalStopHandler>;
    MouseUpDownSignal mouseDownSignal; // signal is called on mouse down
    MouseUpDownSignal mouseUpSignal; // signal is called on mouse up
    MouseMoveSignal mouseMoveSignal; // signal is called on mouse move, note that input x and y are in screen space
    MouseScrollSignal mouseScrollSignal; // signal is called on mouse is scrolled
    // High-level mouse events for clicks and dragging, emitted by MouseController
    // When mouseClickSignal has connections, a small delay for click detection is introduced into camera operations and dragging
    // Dragging starts if dragStartSignal is handled (returns true), and ends on button release
    // When dragging is active, dragSignal and dragEndSignal are emitted instead of mouseMove and mouseUp
    // mouseDown handler have priority over dragStart
    MouseUpDownSignal mouseClickSignal; // signal is called when mouse button is pressed and immediately released
    MouseUpDownSignal dragStartSignal; // signal is called when mouse button is pressed (deferred if click behavior is on)
    MouseUpDownSignal dragEndSignal; // signal is called when mouse button used to start drag is released
    MouseMoveSignal dragSignal; // signal is called when mouse is being dragged with button down
    // Cursor enters/leaves
    using CursorEntranceSignal = boost::signals2::signal<void(bool)>;
    CursorEntranceSignal cursorEntranceSignal;
    // Keyboard event
    using CharPressedSignal = boost::signals2::signal<bool( unsigned unicodeKey, int modifier ), SignalStopHandler>;
    using KeySignal = boost::signals2::signal<bool( int key, int modifier ), SignalStopHandler>;
    CharPressedSignal charPressedSignal; // signal is called when unicode char on/is down/pressed for some time
    KeySignal keyUpSignal; // signal is called on key up
    KeySignal keyDownSignal; // signal is called on key down
    KeySignal keyRepeatSignal; // signal is called when key is pressed for some time
    // SpaceMouseEvents
    using SpaceMouseMoveSignal = boost::signals2::signal<bool( const Vector3f& translate, const Vector3f& rotate ), SignalStopHandler>;
    using SpaceMouseKeySignal = boost::signals2::signal<bool( int ), SignalStopHandler>;
    SpaceMouseMoveSignal spaceMouseMoveSignal; // signal is called on spacemouse 3d controller (joystick) move
    SpaceMouseKeySignal spaceMouseDownSignal; // signal is called on spacemouse key down
    SpaceMouseKeySignal spaceMouseUpSignal; // signal is called on spacemouse key up
    SpaceMouseKeySignal spaceMouseRepeatSignal; // signal is called when spacemouse key is pressed for some time
    // Render events
    using RenderSignal = boost::signals2::signal<void()>;
    RenderSignal preSetupViewSignal; // signal is called before viewports cleanup and camera setup, so one can customize camera XFs for this frame
    RenderSignal preDrawSignal; // signal is called before scene draw (but after scene setup)
    RenderSignal preDrawPostViewportSignal; // signal is called before scene draw but after viewport.preDraw()
    RenderSignal drawSignal; // signal is called on scene draw (after objects tree but before viewport.postDraw())
    RenderSignal postDrawPreViewportSignal; // signal is called after scene draw but after before viewport.postDraw()
    RenderSignal postDrawSignal; // signal is called after scene draw
    // Scene events
    using ObjectsLoadedSignal = boost::signals2::signal<void( const std::vector<std::shared_ptr<Object>>& objs, const std::string& errors, const std::string& warnings )>;
    using DragDropSignal = boost::signals2::signal<bool( const std::vector<std::filesystem::path>& paths ), SignalStopHandler>;
    using PostResizeSignal = boost::signals2::signal<void( int x, int y )>;
    using PostRescaleSignal = boost::signals2::signal<void( float xscale, float yscale )>;
    using InterruptCloseSignal = boost::signals2::signal<bool(), SignalStopHandler>;
    ObjectsLoadedSignal objectsLoadedSignal; // signal is called when objects are loaded by Viewer::loadFiles  function
    CursorEntranceSignal dragEntranceSignal; // signal is called on drag enter/leave the window
    MouseMoveSignal dragOverSignal; // signal is called on drag coordinate changed
    DragDropSignal dragDropSignal; // signal is called on drag and drop file
    PostResizeSignal postResizeSignal; // signal is called after window resize
    PostRescaleSignal postRescaleSignal; // signal is called after window rescale
    InterruptCloseSignal interruptCloseSignal; // signal is called before close window (return true will prevent closing)
    // Touch signals
    using TouchSignal = boost::signals2::signal<bool(int,int,int), SignalStopHandler>;
    TouchSignal touchStartSignal; // signal is called when any touch starts
    TouchSignal touchMoveSignal; // signal is called when touch moves
    TouchSignal touchEndSignal; // signal is called when touch stops
    // Touchpad gesture events
    using TouchpadGestureBeginSignal = boost::signals2::signal<bool(), SignalStopHandler>;
    using TouchpadGestureEndSignal = boost::signals2::signal<bool(), SignalStopHandler>;
    using TouchpadRotateGestureUpdateSignal = boost::signals2::signal<bool( float angle ), SignalStopHandler>;
    using TouchpadSwipeGestureUpdateSignal = boost::signals2::signal<bool( float deltaX, float deltaY, bool kinetic ), SignalStopHandler>;
    using TouchpadZoomGestureUpdateSignal = boost::signals2::signal<bool( float scale, bool kinetic ), SignalStopHandler>;
    TouchpadGestureBeginSignal touchpadRotateGestureBeginSignal; // signal is called on touchpad rotate gesture beginning
    TouchpadRotateGestureUpdateSignal touchpadRotateGestureUpdateSignal; // signal is called on touchpad rotate gesture update
    TouchpadGestureEndSignal touchpadRotateGestureEndSignal; // signal is called on touchpad rotate gesture end
    TouchpadGestureBeginSignal touchpadSwipeGestureBeginSignal; // signal is called on touchpad swipe gesture beginning
    TouchpadSwipeGestureUpdateSignal touchpadSwipeGestureUpdateSignal; // signal is called on touchpad swipe gesture update
    TouchpadGestureEndSignal touchpadSwipeGestureEndSignal; // signal is called on touchpad swipe gesture end
    TouchpadGestureBeginSignal touchpadZoomGestureBeginSignal; // signal is called on touchpad zoom gesture beginning
    TouchpadZoomGestureUpdateSignal touchpadZoomGestureUpdateSignal; // signal is called on touchpad zoom gesture update
    TouchpadGestureEndSignal touchpadZoomGestureEndSignal; // signal is called on touchpad zoom gesture end
    // Window focus signal
    using PostFocusSignal = boost::signals2::signal<void( bool )>;
    PostFocusSignal postFocusSignal;
};

} //namespace MR
