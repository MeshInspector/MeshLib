#pragma once

#include "MRViewerInstance.h"
#include "MRMouse.h"
#include "MRSignalCombiners.h"
#include "MRMakeSlot.h"
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRViewportId.h>
#include "MRMesh/MRSignal.h"
#include "MRMesh/MRRenderModelParameters.h"
#include <cstdint>
#include <filesystem>

struct GLFWwindow;

/// helper macros to add an `MR::Viewer` method call to the event queue
#define ENQUEUE_VIEWER_METHOD( NAME, METHOD ) MR::getViewerInstance().emplaceEvent( NAME, [] { \
    MR::getViewerInstance() . METHOD (); \
} )
#define ENQUEUE_VIEWER_METHOD_ARGS( NAME, METHOD, ... ) MR::getViewerInstance().emplaceEvent( NAME, [__VA_ARGS__] { \
    MR::getViewerInstance() . METHOD ( __VA_ARGS__ ); \
} )
#define ENQUEUE_VIEWER_METHOD_ARGS_SKIPABLE( NAME, METHOD, ... ) MR::getViewerInstance().emplaceEvent( NAME, [__VA_ARGS__] { \
    MR::getViewerInstance() . METHOD ( __VA_ARGS__ ); \
}, true )

namespace MR
{

class ViewerTitle;

class SpaceMouseHandler;

class IDragDropHandler;

class CornerControllerObject;

class ViewportGlobalBasis;

// This struct contains rules for viewer launch
struct LaunchParams
{
    bool fullscreen{ false }; // if true starts fullscreen
    int width{ 0 };
    int height{ 0 };
    enum WindowMode
    {
        Show, // Show window immediately
        HideInit, // Show window after init
        Hide, // Don't show window
        TryHidden, // Launches in "Hide" mode if OpenGL is present and "NoWindow" if it is not
        NoWindow // Don't initialize GL window (don't call GL functions)(force `isAnimating`)
    } windowMode{ HideInit };
    bool enableTransparentBackground{ false };
    bool preferOpenGL3{ false };
    bool render3dSceneInTexture{ true }; // If not set renders scene each frame
    bool developerFeatures{ false }; // If set shows some developer features useful for debugging
    std::string name{ "MRViewer" }; // Window name
    bool startEventLoop{ true }; // If false - does not start event loop
    bool close{ true }; // If !startEventLoop close immediately after start, otherwise close on window close, make sure you call `launchShut` manually if this flag is false
    bool console{ false }; // If true - shows developers console
    int argc{ 0 }; // Pass argc
    char** argv{ nullptr }; // Pass argv

    bool showMRVersionInTitle{ false }; // if true - print version info in window title
    bool isAnimating{ false }; // if true - calls render without system events
    int animationMaxFps{ 30 }; // max fps if animating
    bool unloadPluginsAtEnd{ false }; // unload all extended libraries right before program exit

    std::shared_ptr<SplashWindow> splashWindow; // if present will show this window while initializing plugins (after menu initialization)
};

using FilesLoadedCallback = std::function<void(const std::vector<std::shared_ptr<Object>>& objs,const std::string& errors, const std::string& warnings)>;

struct FileLoadOptions
{
    /// first part of undo name
    const char * undoPrefix = "Open ";

    enum class ReplaceMode
    {
        ContructionBased, ///< replace current scene if new one was loaded from single scene file
        ForceReplace,
        ForceAdd
    };

    /// Determines how to deal with current scene after loading new one
    ReplaceMode replaceMode = ReplaceMode::ContructionBased;

    /// if this callback is set - it is called once when all objects are added to scene
    /// top level objects only are present here
    FilesLoadedCallback loadedCallback;
};

// GLFW-based mesh viewer
class MRVIEWER_CLASS Viewer
{
public:
    using MouseButton = MR::MouseButton;
    using MouseMode = MR::MouseMode;

    using LaunchParams = MR::LaunchParams;

    // Accumulate launch params from cmd args
    MRVIEWER_API static void parseLaunchParams( LaunchParams& params );

    // Launch viewer with given params
    MRVIEWER_API int launch( const LaunchParams& params );
    // Starts event loop
    MRVIEWER_API void launchEventLoop();
    // Terminate window
    MRVIEWER_API void launchShut();

    bool isLaunched() const { return isLaunched_; }

    // get full parameters with witch viewer was launched
    const LaunchParams& getLaunchParams() const { return launchParams_; }

    // provides non const access to viewer
    static Viewer* instance() { return &getViewerInstance(); }
    static Viewer& instanceRef() { return getViewerInstance(); }
    // provide const access to viewer
    static const Viewer* constInstance() { return &getViewerInstance(); }
    static const Viewer& constInstanceRef() { return getViewerInstance(); }

    template<typename PluginType>
    PluginType* getPluginInstance()
    {
        for ( auto& plugin : plugins )
        {
            auto p = dynamic_cast< PluginType* >( plugin );
            if ( p )
            {
                return p;
            }
        }
        return nullptr;
    }

    // Mesh IO
    // Check the supported file format
    MRVIEWER_API bool isSupportedFormat( const std::filesystem::path& file_name );

    // Load objects / scenes from files
    // Note! load files with progress bar in next frame if it possible, otherwise load directly inside this function
    MRVIEWER_API bool loadFiles( const std::vector< std::filesystem::path>& filesList, const FileLoadOptions & options = {} );

    // Save first selected objects to file
    MRVIEWER_API bool saveToFile( const std::filesystem::path & mesh_file_name );

    // Callbacks
    MRVIEWER_API bool keyPressed( unsigned int unicode_key, int modifier );
    MRVIEWER_API bool keyDown( int key, int modifier );
    MRVIEWER_API bool keyUp( int key, int modifier );
    MRVIEWER_API bool keyRepeat( int key, int modifier );
    MRVIEWER_API bool mouseDown( MouseButton button, int modifier );
    MRVIEWER_API bool mouseUp( MouseButton button, int modifier );
    MRVIEWER_API bool mouseMove( int mouse_x, int mouse_y );
    MRVIEWER_API bool mouseScroll( float delta_y );
    MRVIEWER_API bool mouseClick( MouseButton button, int modifier );
    MRVIEWER_API bool dragStart( MouseButton button, int modifier );
    MRVIEWER_API bool dragEnd( MouseButton button, int modifier );
    MRVIEWER_API bool drag( int mouse_x, int mouse_y );
    MRVIEWER_API bool spaceMouseMove( const Vector3f& translate, const Vector3f& rotate );
    MRVIEWER_API bool spaceMouseDown( int key );
    MRVIEWER_API bool spaceMouseUp( int key );
    MRVIEWER_API bool spaceMouseRepeat( int key );
    MRVIEWER_API bool dragDrop( const std::vector<std::filesystem::path>& paths  );
    // Touch callbacks (now used in EMSCRIPTEN build only)
    MRVIEWER_API bool touchStart( int id, int x, int y );
    MRVIEWER_API bool touchMove( int id, int x, int y );
    MRVIEWER_API bool touchEnd( int id, int x, int y );
    // Touchpad gesture callbacks
    MRVIEWER_API bool touchpadRotateGestureBegin();
    MRVIEWER_API bool touchpadRotateGestureUpdate( float angle );
    MRVIEWER_API bool touchpadRotateGestureEnd();
    MRVIEWER_API bool touchpadSwipeGestureBegin();
    MRVIEWER_API bool touchpadSwipeGestureUpdate( float dx, float dy, bool kinetic );
    MRVIEWER_API bool touchpadSwipeGestureEnd();
    MRVIEWER_API bool touchpadZoomGestureBegin();
    MRVIEWER_API bool touchpadZoomGestureUpdate( float scale, bool kinetic );
    MRVIEWER_API bool touchpadZoomGestureEnd();
    // This function is called when window should close, if return value is true, window will stay open
    MRVIEWER_API bool interruptWindowClose();

    // Draw everything
    MRVIEWER_API void draw( bool force = false );
    // Draw 3d scene with UI
    MRVIEWER_API void drawFull( bool dirtyScene );
    // Draw 3d scene without UI
    MRVIEWER_API void drawScene();
    // Call this function to force redraw scene into scene texture
    void setSceneDirty() { dirtyScene_ = true; }
    // Setup viewports views
    MRVIEWER_API void setupScene();
    // Cleans framebuffers for all viewports (sets its background)
    MRVIEWER_API void clearFramebuffers();
    // OpenGL context resize
    MRVIEWER_API void resize( int w, int h ); // explicitly set framebuffer size
    MRVIEWER_API void postResize( int w, int h ); // external resize due to user interaction
    MRVIEWER_API void postSetPosition( int xPos, int yPos ); // external set position due to user interaction
    MRVIEWER_API void postSetMaximized( bool maximized ); // external set maximized due to user interaction
    MRVIEWER_API void postSetIconified( bool iconified ); // external set iconified due to user interaction
    MRVIEWER_API void postFocus( bool focused ); // external focus handler due to user interaction
    MRVIEWER_API void postRescale( float x, float y ); // external rescale due to user interaction
    MRVIEWER_API void postClose(); // called when close signal received

    ////////////////////////
    // Multi-mesh methods //
    ////////////////////////

    // reset objectRoot with newRoot, append all RenderObjects and basis objects
    MRVIEWER_API void set_root( SceneRootObject& newRoot );

    // removes all objects from scene
    MRVIEWER_API void clearScene();

    ////////////////////////////
    // Multi-viewport methods //
    ////////////////////////////

    // Return the current viewport, or the viewport corresponding to a given unique identifier
    //
    // Inputs:
    //   viewportId unique identifier corresponding to the desired viewport (current viewport if 0)
    MRVIEWER_API Viewport& viewport( ViewportId viewportId = {} );
    MRVIEWER_API const Viewport& viewport( ViewportId viewportId = {} ) const;

    // Append a new "slot" for a viewport (i.e., copy properties of the current viewport, only
    // changing the viewport size/position)
    //
    // Inputs:
    //   viewport      Vector specifying the viewport origin and size in screen coordinates.
    //   append_empty  If true, existing meshes are hidden on the new viewport.
    //
    // Returns the unique id of the newly inserted viewport. There can be a maximum of 31
    //   viewports created in the same viewport. Erasing a viewport does not change the id of
    //   other existing viewports
    MRVIEWER_API ViewportId append_viewport( const ViewportRectangle & viewportRect, bool append_empty = false );

    // Calculates and returns viewports bounds in gl space:
    // (0,0) - lower left angle
    MRVIEWER_API Box2f getViewportsBounds() const;

    // Erase a viewport
    //
    // Inputs:
    //   index  index of the viewport to erase
    MRVIEWER_API bool erase_viewport( const size_t index );
    MRVIEWER_API bool erase_viewport( ViewportId viewport_id );

    // Retrieve viewport index from its unique identifier
    // Returns -1 if not found
    MRVIEWER_API int viewport_index( ViewportId viewport_id ) const;

    // Get unique id of the vieport containing the mouse
    // if mouse is out of any viewport returns index of last selected viewport
    // (current_mouse_x, current_mouse_y)
    MRVIEWER_API ViewportId getHoveredViewportId() const;

    // Change selected_core_index to the viewport containing the mouse
    // (current_mouse_x, current_mouse_y)
    MRVIEWER_API void select_hovered_viewport();

    // Calls fitData for single/each viewport in viewer
    // fill = 0.6 parameter means that scene will 0.6 of screen,
    // snapView - to snap camera angle to closest canonical quaternion
    MRVIEWER_API void fitDataViewport( MR::ViewportMask vpList = MR::ViewportMask::all(), float fill = 0.6f, bool snapView = true );

    // Calls fitBox for single/each viewport in viewer
    // fill = 0.6 parameter means that scene will 0.6 of screen,
    // snapView - to snap camera angle to closest canonical quaternion
    MRVIEWER_API void fitBoxViewport( const Box3f& box, MR::ViewportMask vpList = MR::ViewportMask::all(), float fill = 0.6f, bool snapView = true );

    // Calls fitData and change FOV to match the screen size then
    // params - params fit data
    MRVIEWER_API void preciseFitDataViewport( MR::ViewportMask vpList = MR::ViewportMask::all() );
    MRVIEWER_API void preciseFitDataViewport( MR::ViewportMask vpList, const FitDataParams& param );

    MRVIEWER_API size_t getTotalFrames() const;
    MRVIEWER_API size_t getSwappedFrames() const;
    MRVIEWER_API size_t getFPS() const;
    MRVIEWER_API double getPrevFrameDrawTimeMillisec() const;

    // Returns memory amount used by shared GL memory buffer
    MRVIEWER_API size_t getStaticGLBufferSize() const;

    // if true only last frame of force redraw after events will be swapped, otherwise each will be swapped
    bool swapOnLastPostEventsRedraw{ true };
    // minimum auto increment force redraw frames after events
    int forceRedrawMinimumIncrementAfterEvents{ 4 };

    // Increment number of forced frames to redraw in event loop
    // if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not
    MRVIEWER_API void incrementForceRedrawFrames( int i = 1, bool swapOnLastOnly = false );

    // Returns true if current frame will be shown on display
    MRVIEWER_API bool isCurrentFrameSwapping() const;

    // types of counted events
    enum class EventType
    {
        MouseDown,
        MouseUp,
        MouseMove,
        MouseScroll,
        KeyDown,
        KeyUp,
        KeyRepeat,
        CharPressed,
        Count
    };
    // Returns number of events of given type
    MRVIEWER_API size_t getEventsCount( EventType type )const;

    // types of gl primitives counters
    enum class GLPrimitivesType
    {
        // arrays and elements are different gl calls
        PointArraySize,
        LineArraySize,
        TriangleArraySize,
        PointElementsNum,
        LineElementsNum,
        TriangleElementsNum,
        Count
    };
    // Returns number of events of given type
    MRVIEWER_API size_t getLastFrameGLPrimitivesCount( GLPrimitivesType type ) const;
    // Increment number of gl primitives drawed in this frame
    MRVIEWER_API void incrementThisFrameGLPrimitivesCount( GLPrimitivesType type, size_t num );


    // Returns mask of present viewports
    ViewportMask getPresentViewports() const { return presentViewportsMask_; }

    // Restes frames counter and events counter
    MRVIEWER_API void resetAllCounters();

    /**
     * Captures 3d scene
     * @param resolution resolution of the image <= 0 means default
     */
    MRVIEWER_API Image captureSceneScreenShot( const Vector2i& resolution = Vector2i() );

    /**
     * Captures part of window in the beginning of next frame, capturing all that was drawn in this frame
     * @param callback will be called right when screenshot is taken
     * @param pos left-bottom corner of capturing area relative of left-down corner of window. default = size(0, 0)
     * @param size size of capturing area. default = size(0, 0) - auto size to right-top corner of window.
     */
    MRVIEWER_API void captureUIScreenShot( std::function<void( const Image& )> callback,
                                           const Vector2i& pos = Vector2i(), const Vector2i& size = Vector2i() );

    // Returns true if can enable alpha sort
    MRVIEWER_API bool isAlphaSortAvailable() const;
    // Tries to enable alpha sort,
    // returns true if value was changed, return false otherwise
    MRVIEWER_API bool enableAlphaSort( bool on );
    // Returns true if alpha sort is enabled, false otherwise
    bool isAlphaSortEnabled() const { return alphaSortEnabled_; }

    // Returns if scene texture is now bound
    MRVIEWER_API bool isSceneTextureBound()  const;
    // Binds or unbinds scene texture (should be called only with valid window)
    // note that it does not clear framebuffer
    MRVIEWER_API void bindSceneTexture( bool bind );
    // Returns true if 3d scene is rendering in scene texture instead of main framebuffer
    MRVIEWER_API bool isSceneTextureEnabled() const;

    // Returns actual msaa level of:
    //  scene texture if it is present, or main framebuffer
    MRVIEWER_API int getMSAA() const;
    // Requests changing MSAA level
    // if scene texture is using, request should be executed in the beginig of next frame
    // otherwise restart of the app is required to apply change to main framebuffer
    MRVIEWER_API void requestChangeMSAA( int newMSAA );
    // Returns MSAA level that have been requested (might be different from actual MSAA using, because of GPU limitations or need to restart app)
    MRVIEWER_API int getRequestedMSAA() const;

    // Sets manager of viewer settings which loads user personal settings on beginning of app
    // and saves it in app's ending
    MRVIEWER_API void setViewportSettingsManager( std::unique_ptr<IViewerSettingsManager> mng );
    MRVIEWER_API const std::unique_ptr<IViewerSettingsManager>& getViewerSettingsManager() const { return settingsMng_; }

    using PointInAllSpaces = MR::PointInAllSpaces;
    // Finds point in all spaces from screen space pixel point
    MRVIEWER_API PointInAllSpaces getPixelPointInfo( const Vector3f& screenPoint ) const;
    // Finds point under mouse in all spaces and under mouse viewport id
    MRVIEWER_API PointInAllSpaces getMousePointInfo() const;

    // Converts screen space coordinate to viewport space coordinate
    // (0,0) if viewport does not exist
    // screen space: X [0,framebufferSize.x], Y [0,framebufferSize.y] - (0,0) is upper left of window
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0,1] - 0 is Dnear, 1 is Dfar
    MRVIEWER_API Vector3f screenToViewport( const Vector3f& screenPoint, ViewportId id ) const;
    // Converts viewport space coordinate to screen space coordinate
    // (0,0) if viewport does not exist
    // screen space: X [0,framebufferSize.x], Y [0,framebufferSize.y] - (0,0) is upper left of window
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0,1] - 0 is Dnear, 1 is Dfar
    MRVIEWER_API Vector3f viewportToScreen( const Vector3f& viewportPoint, ViewportId id ) const;

    // Returns viewports satisfying given mask
    MRVIEWER_API std::vector<std::reference_wrapper<Viewport>> getViewports( ViewportMask mask = ViewportMask::any() );

    // Enables or disables global history (clears it on disable)
    MRVIEWER_API void enableGlobalHistory( bool on );
    // Return true if global history is enabled, false otherwise
    bool isGlobalHistoryEnabled() const { return bool( globalHistoryStore_ ); };
    // Appends history action to current stack position (clearing redo)
    // if global history is disabled do nothing
    MRVIEWER_API void appendHistoryAction( const std::shared_ptr<HistoryAction>& action );
    // Applies undo if global history is enabled
    // return true if undo was applied
    MRVIEWER_API bool globalHistoryUndo();
    // Applies redo if global history is enabled
    // return true if redo was applied
    MRVIEWER_API bool globalHistoryRedo();
    // Returns global history store
    const std::shared_ptr<HistoryStore>& getGlobalHistoryStore() const { return globalHistoryStore_; }
    // Return spacemouse handler
    const std::shared_ptr<SpaceMouseHandler>& getSpaceMouseHandler() const { return spaceMouseHandler_; }

    // This method is called after successful scene saving to update scene root, window title and undo
    MRVIEWER_API void onSceneSaved( const std::filesystem::path& savePath, bool storeInRecent = true );

    // Get/Set menu plugin (which is separated from other plugins to be inited first before splash window starts)
    MRVIEWER_API const std::shared_ptr<ImGuiMenu>& getMenuPlugin() const;
    MRVIEWER_API void setMenuPlugin( std::shared_ptr<ImGuiMenu> menu );

    // get menu plugin casted in RibbonMenu
    MRVIEWER_API std::shared_ptr<RibbonMenu> getRibbonMenu() const;

    // Get the menu plugin casted in given type
    template <typename T>
    std::shared_ptr<T> getMenuPluginAs() const { return std::dynamic_pointer_cast<T>( getMenuPlugin() ); }

    // sets stop event loop flag (this flag is glfwShouldWindowClose equivalent)
    MRVIEWER_API void stopEventLoop();
    // get stop event loop flag (this flag is glfwShouldWindowClose equivalent)
    bool getStopEventLoopFlag() const { return stopEventLoop_; }

    // return true if window should close
    // calls interrupt signal and if no slot interrupts return true, otherwise return false
    bool windowShouldClose();

    // returns true if viewer has valid GL context
    // note that sometimes it is not enough, for example to free GL memory in destructor,
    // glInitialized_ can be already reset and it requires `loadGL()` check too
    bool isGLInitialized() const { return glInitialized_; }

    // update the title of the main window and, if any scene was opened, show its filename
    MRVIEWER_API void makeTitleFromSceneRootPath();

    // returns true if the system framebuffer is scaled (valid for macOS and Wayland)
    bool hasScaledFramebuffer() const { return hasScaledFramebuffer_; }

public:
    //////////////////////
    // Member variables //
    //////////////////////
    GLFWwindow* window;

    // A function to reset setting to initial state
    // Overrides should call previous function
    std::function<void( Viewer* viewer )> resetSettingsFunction;

    // Stores all the viewing options
    std::vector<Viewport> viewport_list;
    size_t selected_viewport_index;

    // List of registered plugins
    std::vector<ViewerPlugin*> plugins;

    float pixelRatio{ 1.0f };
    Vector2i framebufferSize;
    Vector2i windowSavePos; // pos to save
    Vector2i windowSaveSize; // size to save
    Vector2i windowOldPos;
    bool windowMaximized{ false };

    // if true - calls render without system events
    bool isAnimating{ false };
    // max fps if animating
    int animationMaxFps{ 30 };
    // this parameter can force up/down mouse scroll
    // useful for WebAssembler version because it has too powerful scroll
    float scrollForce{ }; // init in resetSettingsFunction()
    // opengl-based pick window radius in pixels
    uint16_t glPickRadius{ }; // init in resetSettingsFunction()
    // Experimental/developer features enabled
    bool experimentalFeatures{ };
    // command arguments, each parsed arg should be erased from here not to affect other parsers
    std::vector<std::string> commandArgs;

    std::shared_ptr<ObjectMesh> basisAxes;
    std::unique_ptr<CornerControllerObject> basisViewController;
    std::unique_ptr<ViewportGlobalBasis> globalBasis;
    std::shared_ptr<ObjectMesh> rotationSphere;
    // Stores clipping plane mesh
    std::shared_ptr<ObjectMesh> clippingPlaneObject;

    // class that updates viewer title
    std::shared_ptr<ViewerTitle> windowTitle;

    //*********
    // SIGNALS
    //*********
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
    MouseUpDownSignal dragStartSignal; // signal is called when mouse button is pressed (deterred if click behavior is on)
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

    /// emplace event at the end of the queue
    /// replace last skipable with new skipable
    MRVIEWER_API void emplaceEvent( std::string name, ViewerEventCallback cb, bool skipable = false );
    // pop all events from the queue while they have this name
    MRVIEWER_API void popEventByName( const std::string& name );

    MRVIEWER_API void postEmptyEvent();

    [[nodiscard]] MRVIEWER_API const TouchpadParameters & getTouchpadParameters() const;
    MRVIEWER_API void setTouchpadParameters( const TouchpadParameters & );

    [[nodiscard]] MRVIEWER_API SpaceMouseParameters getSpaceMouseParameters() const;
    MRVIEWER_API void setSpaceMouseParameters( const SpaceMouseParameters & );

    [[nodiscard]] const MouseController &mouseController() const { return *mouseController_; }
    [[nodiscard]] MouseController &mouseController() { return *mouseController_; }

    // Store of recently opened files
    [[nodiscard]] const RecentFilesStore &recentFilesStore() const { return *recentFilesStore_; }
    [[nodiscard]] RecentFilesStore &recentFilesStore() { return *recentFilesStore_; }

    /// returns whether to sort the filenames received from Drag&Drop in lexicographical order before adding them in scene
    [[nodiscard]] bool getSortDroppedFiles() const { return sortDroppedFiles_; }

    /// sets whether to sort the filenames received from Drag&Drop in lexicographical order before adding them in scene
    void setSortDroppedFiles( bool value ) { sortDroppedFiles_ = value; }

    /// (re)initializes the handler of SpaceMouse events
    /// \param deviceSignal every device-related event will be sent here: find, connect, disconnect
    MRVIEWER_API void initSpaceMouseHandler( std::function<void(const std::string&)> deviceSignal = {} );

private:
    Viewer();
    ~Viewer();

    // Init window
    int launchInit_( const LaunchParams& params );

    // Called from launchInit_ after window creating to configure it properly
    bool setupWindow_( const LaunchParams& params );

    // Return true if OpenGL loaded successfully
    bool checkOpenGL_(const LaunchParams& params );

    // Init base objects
    void init_();

    // Init all plugins on start
    void initPlugins_();

    // Shut all plugins at the end
    void shutdownPlugins_();

#ifdef __EMSCRIPTEN__
    void mainLoopFunc_();
    static void emsMainInfiniteLoop();
#endif
    // returns true if was swapped
    bool draw_( bool force );

    void drawUiRenderObjects_();

    // the minimum number of frames to be rendered even if the scene is unchanged
    int forceRedrawFrames_{ 0 };
    // Should be `<= forceRedrawFrames_`. The next N frames will not be shown on screen.
    int forceRedrawFramesWithoutSwap_{ 0 };

    // if this flag is set shows some developer features useful for debugging
    bool enableDeveloperFeatures_{ false };

    std::unique_ptr<ViewerEventQueue> eventQueue_;

    // special plugin for menu (initialized before splash window starts)
    std::shared_ptr<ImGuiMenu> menuPlugin_;

    std::unique_ptr<TouchpadController> touchpadController_;
    std::unique_ptr<SpaceMouseController> spaceMouseController_;
    std::unique_ptr<TouchesController> touchesController_;
    std::unique_ptr<MouseController> mouseController_;
    std::unique_ptr<IDragDropHandler> dragDropAdvancedHandler_;

    std::unique_ptr<RecentFilesStore> recentFilesStore_;
    std::unique_ptr<FrameCounter> frameCounter_;

    mutable struct EventsCounter
    {
        std::array<size_t, size_t( EventType::Count )> counter{};
        void reset();
    } eventsCounter_;

    mutable struct GLPrimitivesCounter
    {
        std::array<size_t, size_t( GLPrimitivesType::Count )> counter{};
        void reset();
    } glPrimitivesCounter_;


    // creates glfw window with gl version major.minor, false if failed;
    bool tryCreateWindow_( bool fullscreen, int& width, int& height, const std::string& name, int major, int minor );

    bool needRedraw_() const;
    void resetRedraw_();

    void recursiveDraw_( const Viewport& vp, const Object& obj, const AffineXf3f& parentXf, RenderModelPassMask renderType, int* numDraws = nullptr ) const;

    void initGlobalBasisAxesObject_();
    void initBasisAxesObject_();
    void initBasisViewControllerObject_();
    void initClippingPlaneObject_();
    void initRotationCenterObject_();

    // recalculate pixel ratio
    void updatePixelRatio_();

    // return MSAA that is required for framebuffer
    // sceneTextureOn - true means that app is using scene texture for rendering (false means that scene is rendered directly in main framebuffer)
    // forSceneTexture - true request MSAA required for scene texture (calling with !sceneTextureOn is invalid), false - request MSAA for main framebuffer
    int getRequiredMSAA_( bool sceneTextureOn, bool forSceneTexture ) const;

    bool stopEventLoop_{ false };

    bool isLaunched_{ false };
    // this flag is needed to know if all viewer setup was already done, and we can call draw
    bool focusRedrawReady_{ false };

    std::unique_ptr<SceneTextureGL> sceneTexture_;
    std::unique_ptr<AlphaSortGL> alphaSorter_;

    bool alphaSortEnabled_{false};

    bool glInitialized_{ false };

    bool isInDraw_{ false };
    bool dirtyScene_{ false };

    bool hasScaledFramebuffer_{ false };

    bool sortDroppedFiles_{ true };

    LaunchParams launchParams_;

    ViewportId getFirstAvailableViewportId_() const;
    ViewportMask presentViewportsMask_;

    std::unique_ptr<IViewerSettingsManager> settingsMng_;

    std::shared_ptr<HistoryStore> globalHistoryStore_;

    std::shared_ptr<SpaceMouseHandler> spaceMouseHandler_;

    std::vector<boost::signals2::scoped_connection> uiUpdateConnections_;

    friend MRVIEWER_API Viewer& getViewerInstance();
};

// starts default viewer with given params and setup
MRVIEWER_API int launchDefaultViewer( const Viewer::LaunchParams& params, const ViewerSetup& setup );

// call this function to load MRViewer.dll
MRVIEWER_API void loadMRViewerDll();

} // end namespace
