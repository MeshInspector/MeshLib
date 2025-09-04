#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRSignal.h"
#include <filesystem>
#include <vector>

namespace MR
{

struct IConnectable
{
    virtual ~IConnectable() = default;
    virtual void connect( Viewer* viewer, int group = 0, boost::signals2::connect_position pos = boost::signals2::connect_position::at_back ) = 0;
    virtual void disconnect() = 0;
};

struct ConnectionHolder : virtual IConnectable
{
    virtual ~ConnectionHolder() = default;
    virtual void disconnect() { connection_.disconnect(); }
protected:
    boost::signals2::scoped_connection connection_;
};

template<typename ...Connectables>
struct MultiListener : Connectables...
{
    static_assert( ( std::is_base_of_v<IConnectable, Connectables> && ... ),
        "Base classes must be children of IConnectable" );

    virtual ~MultiListener() = default;

    virtual void connect( 
        [[maybe_unused]] Viewer* viewer, // unused if Connectables is empty
        [[maybe_unused]] int group = 0,
        [[maybe_unused]] boost::signals2::connect_position pos = boost::signals2::connect_position::at_back )
    {
        ( Connectables::connect( viewer, group, pos ), ... );
    }
    virtual void disconnect()
    {
        // disconnect in reversed order
        [[maybe_unused]] int dummy;
        (void)( dummy = ... = ( Connectables::disconnect(), 0 ) );
    }
};

struct MRVIEWER_CLASS MouseDownListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( MouseDownListener );
    virtual ~MouseDownListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onMouseDown_( MouseButton btn, int modifiers ) = 0;
};

struct MRVIEWER_CLASS MouseUpListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( MouseUpListener );
    virtual ~MouseUpListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onMouseUp_( MouseButton btn, int modifiers ) = 0;
};

struct MRVIEWER_CLASS MouseMoveListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( MouseMoveListener );
    virtual ~MouseMoveListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onMouseMove_( int x, int y ) = 0;
};

struct MRVIEWER_CLASS MouseScrollListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( MouseScrollListener );
    virtual ~MouseScrollListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onMouseScroll_( float delta ) = 0;
};

struct MRVIEWER_CLASS MouseClickListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( MouseClickListener );
    virtual ~MouseClickListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onMouseClick_( MouseButton btn, int modifiers ) = 0;
};

struct MRVIEWER_CLASS DragStartListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragStartListener );
    virtual ~DragStartListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onDragStart_( MouseButton btn, int modifiers ) = 0;
};

struct MRVIEWER_CLASS DragEndListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragEndListener );
    virtual ~DragEndListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onDragEnd_( MouseButton btn, int modifiers ) = 0;
};

struct MRVIEWER_CLASS DragListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragListener );
    virtual ~DragListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onDrag_( int x, int y ) = 0;
};

struct MRVIEWER_CLASS CharPressedListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( CharPressedListener );
    virtual ~CharPressedListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onCharPressed_( unsigned charKey, int modifier ) = 0;
};

struct MRVIEWER_CLASS KeyUpListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( KeyUpListener );
    virtual ~KeyUpListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onKeyUp_( int key, int modifier ) = 0;
};

struct MRVIEWER_CLASS KeyDownListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( KeyDownListener );
    virtual ~KeyDownListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onKeyDown_( int key, int modifier ) = 0;
};

struct MRVIEWER_CLASS KeyRepeatListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( KeyRepeatListener );
    virtual ~KeyRepeatListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onKeyRepeat_( int key, int modifier ) = 0;
};

struct MRVIEWER_CLASS PreSetupViewListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PreSetupViewListener );
    virtual ~PreSetupViewListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void preSetupView_() = 0;
};

struct MRVIEWER_CLASS PreDrawListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PreDrawListener );
    virtual ~PreDrawListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void preDraw_() = 0;
};

struct MRVIEWER_CLASS DrawListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DrawListener );
    virtual ~DrawListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void draw_() = 0;
};

struct MRVIEWER_CLASS PostDrawListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PostDrawListener );
    virtual ~PostDrawListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void postDraw_() = 0;
};

struct MRVIEWER_CLASS DragDropListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragDropListener );
    virtual ~DragDropListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool dragDrop_( const std::vector<std::filesystem::path>& paths ) = 0;
};

struct MRVIEWER_CLASS DragEntranceListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragEntranceListener );
    virtual ~DragEntranceListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void dragEntrance_( bool enter ) = 0;
};

struct MRVIEWER_CLASS DragOverListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DragOverListener );
    virtual ~DragOverListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool dragOver_( int x, int y ) = 0;
};

struct MRVIEWER_CLASS PostResizeListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PostResizeListener );
    virtual ~PostResizeListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void postResize_( int w, int h ) = 0;
};

struct MRVIEWER_CLASS InterruptCloseListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( InterruptCloseListener );
    virtual ~InterruptCloseListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool interruptClose_() = 0;
};

struct MRVIEWER_CLASS PostRescaleListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PostRescaleListener );
    virtual ~PostRescaleListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void postRescale_( float x, float y ) = 0;
};

struct MRVIEWER_CLASS TouchStartListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchStartListener );
    virtual ~TouchStartListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onTouchStart_( int id, int x, int y ) = 0;
};

struct MRVIEWER_CLASS TouchMoveListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchMoveListener );
    virtual ~TouchMoveListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onTouchMove_( int id, int x, int y ) = 0;
};

struct MRVIEWER_CLASS TouchEndListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchEndListener );
    virtual ~TouchEndListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onTouchEnd_( int id, int x, int y ) = 0;
};

/// class to subscribe on SpaceMouseMoveSignal
struct MRVIEWER_CLASS SpaceMouseMoveListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( SpaceMouseMoveListener );
    virtual ~SpaceMouseMoveListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool spaceMouseMove_( const Vector3f& translate, const Vector3f& rotate ) = 0;
};

/// class to subscribe on SpaceMouseDownSgnal
struct MRVIEWER_CLASS SpaceMouseDownListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( SpaceMouseDownListener );
    virtual ~SpaceMouseDownListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool spaceMouseDown_( int key ) = 0;
};

/// class to subscribe on SpaceMouseUpSignal
struct MRVIEWER_CLASS SpaceMouseUpListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( SpaceMouseUpListener );
    virtual ~SpaceMouseUpListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool spaceMouseUp_( int key ) = 0;
};

/// class to subscribe on TouchpadRotateGestureBeginEvent
struct MRVIEWER_CLASS TouchpadRotateGestureBeginListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadRotateGestureBeginListener );
    virtual ~TouchpadRotateGestureBeginListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadRotateGestureBegin_() = 0;
};

/// class to subscribe on TouchpadRotateGestureUpdateEvent
struct MRVIEWER_CLASS TouchpadRotateGestureUpdateListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadRotateGestureUpdateListener );
    virtual ~TouchpadRotateGestureUpdateListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadRotateGestureUpdate_( float angle ) = 0;
};

/// class to subscribe on TouchpadRotateGestureEndEvent
struct MRVIEWER_CLASS TouchpadRotateGestureEndListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadRotateGestureEndListener );
    virtual ~TouchpadRotateGestureEndListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadRotateGestureEnd_() = 0;
};

/// class to subscribe on TouchpadSwipeGestureBeginEvent
struct MRVIEWER_CLASS TouchpadSwipeGestureBeginListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadSwipeGestureBeginListener );
    virtual ~TouchpadSwipeGestureBeginListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadSwipeGestureBegin_() = 0;
};

/// class to subscribe on TouchpadSwipeGestureUpdateEvent
struct MRVIEWER_CLASS TouchpadSwipeGestureUpdateListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadSwipeGestureUpdateListener );
    virtual ~TouchpadSwipeGestureUpdateListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadSwipeGestureUpdate_( float dx, float dy, bool kinetic ) = 0;
};

/// class to subscribe on TouchpadSwipeGestureEndEvent
struct MRVIEWER_CLASS TouchpadSwipeGestureEndListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadSwipeGestureEndListener );
    virtual ~TouchpadSwipeGestureEndListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadSwipeGestureEnd_() = 0;
};

/// class to subscribe on TouchpadZoomGestureBeginEvent
struct MRVIEWER_CLASS TouchpadZoomGestureBeginListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadZoomGestureBeginListener );
    virtual ~TouchpadZoomGestureBeginListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadZoomGestureBegin_() = 0;
};

/// class to subscribe on TouchpadZoomGestureUpdateEvent
struct MRVIEWER_CLASS TouchpadZoomGestureUpdateListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadZoomGestureUpdateListener );
    virtual ~TouchpadZoomGestureUpdateListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadZoomGestureUpdate_( float scale, bool kinetic ) = 0;
};

/// class to subscribe on TouchpadZoomGestureEndEvent
struct MRVIEWER_CLASS TouchpadZoomGestureEndListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( TouchpadZoomGestureEndListener );
    virtual ~TouchpadZoomGestureEndListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool touchpadZoomGestureEnd_() = 0;
};

/// class to subscribe on PostFocusSingal
struct MRVIEWER_CLASS PostFocusListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( PostFocusListener );
    virtual ~PostFocusListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void postFocus_( bool focused ) = 0;
};

/// class to subscribe on CursorEntranceSingal
struct MRVIEWER_CLASS CursorEntranceListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( CursorEntranceListener );
    virtual ~CursorEntranceListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void cursorEntrance_( bool enter ) = 0;
};

}
