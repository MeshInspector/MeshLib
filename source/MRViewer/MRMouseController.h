#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRphmap.h"
#include "MRMouse.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include <optional>

namespace MR
{
// this class stores two maps:
// 1) mouse control to mouse mode
// 2) mouse mode to mouse control
// it is present as field in Viewer, and used to control scene 
// note: default state is usually set from ViewerSetup class
// note: user config saves its state
class MRVIEWER_CLASS MouseController
{
public:
    MR_ADD_CTOR_DELETE_MOVE( MouseController );
    struct MouseControlKey
    {
        MouseButton btn{ MouseButton::Left };
        int mod{ 0 }; // modifier (alt/ctrl/shift etc.)
    };

    // called in Viewer init, connects to Viewer mouse signals
    MRVIEWER_API void connect();

    // set control
    // note: one control can have only one mode, one mode can have only one control
    // if mode already has other control, other one will be removed
    MRVIEWER_API void setMouseControl( const MouseControlKey& key, MouseMode mode );

    // returns previous mouse down (if several mouse buttons are down returns position of first one)
    const Vector2i& getDownMousePos() const { return downMousePos_; }
    // returns current mouse position
    const Vector2i& getMousePos() const { return currentMousePos_; }
    // returns state of mouse button
    MRVIEWER_API bool isPressed( MouseButton button ) const;

    bool isCursorInside() const { return isCursorInside_; }    

    // returns nullopt if no control is present for given mode, otherwise returns associated control
    MRVIEWER_API std::optional<MouseControlKey> findControlByMode( MouseMode mode ) const;
    // make string from mouse button and modifier
    MRVIEWER_API static std::string getControlString( const MouseControlKey& key );

    // cast mouse button and modifier to simple int key
    MRVIEWER_API static int mouseAndModToKey( const MouseControlKey& key );
    // cast simple int key to mouse button and modifier
    MRVIEWER_API static MouseControlKey keyToMouseAndMod( int key );
private:

#ifdef __EMSCRIPTEN__
    // this is needed to reset previous modes of multitouch events
    void resetAll_();
#endif

    bool preMouseDown_( MouseButton button, int modifier );
    bool mouseDown_( MouseButton button, int modifier );
    bool preMouseUp_( MouseButton button, int modifier );
    bool preMouseMove_( int x, int y );
    bool mouseScroll_( float delta );

    bool isCursorInside_{ false };
    void cursorEntrance_( bool entered );

    Vector3f downTranslation_;
    // screen space
    Vector2i downMousePos_;
    Vector2i prevMousePos_;
    Vector2i currentMousePos_;

    BitSet downState_;
    MouseMode currentMode_{ MouseMode::None };

    using MouseModeMap = HashMap<int, MouseMode>;
    using MouseModeBackMap = HashMap<MouseMode, int>;

    MouseModeMap map_;
    MouseModeBackMap backMap_;
};

}