#pragma once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#include "MRViewport.h"
#include "MRViewerInstance.h"
#include "MRRecentFilesStore.h"
#include "MRMouse.h"
#include <MRMesh/MRObject.h>
#include <MRMesh/MRSceneRoot.h>
#include "MRMesh/MRImage.h"
#include "MRMouseController.h"
#include <boost/signals2/signal.hpp>
#include <cstdint>
#include <queue>

struct GLFWwindow;

template<typename MemberFuncPtr, typename BaseClass>
auto bindSlotCallback( BaseClass* base, MemberFuncPtr func )
{
    static_assert( !( std::is_move_assignable_v<BaseClass> || std::is_move_constructible_v<BaseClass> ), 
                   "MAKE_SLOT requires a non-movable type" );
    return[base, func] ( auto&&... args )
    {
        return ( base->*func )( std::forward<decltype( args )>( args )... );
    };
}

// you will not be able to move your struct after using this macro
#define MAKE_SLOT(func) bindSlotCallback(this,func)

namespace MR
{

class SpaceMouseHandler;

// GLFW-based mesh viewer
class Viewer
{
public:
    using MouseButton = MR::MouseButton;
    using MouseMode = MR::MouseMode;

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
        bool developerFeatures{ false }; // If set shows some developer features useful for debugging
        std::string name{"MRViewer"}; // Window name
        bool startEventLoop{ true }; // If false - does not start event loop
        bool close{ true }; // If !startEventLoop close immediately after start, otherwise close on window close, make sure you call `launchShut` manually if this flag is false
        bool console{ false }; // If true - shows developers console
        int argc{ 0 }; // Pass argc
        char** argv{ nullptr }; // Pass argv

        bool showMRVersionInTitle{ false }; // if true - print version info in window title
        bool isAnimating{ false }; // if true - calls render without system events
        int animationMaxFps{ 30 }; // max fps if animating

        std::shared_ptr<SplashWindow> splashWindow; // if present will show this window while initializing plugins (after menu initialization)
    };

    // Accumulate launch params from cmd args
    MRVIEWER_API static void parseLaunchParams( LaunchParams& params );

    // Launch viewer with given params
    MRVIEWER_API int launch( const LaunchParams& params );
    // Starts event loop
    MRVIEWER_API void launchEventLoop();
    // Terminate window
    MRVIEWER_API void launchShut();
    
    bool isLaunched() const { return isLaunched_; }

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
    MRVIEWER_API bool loadFiles( const std::vector< std::filesystem::path>& filesList );
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
    MRVIEWER_API bool spaceMouseMove( Vector3f translate, Vector3f rotate );
    MRVIEWER_API bool spaceMouseDown( int key );
    MRVIEWER_API bool spaceMouseUp( int key );
    MRVIEWER_API bool dragDrop( const std::vector<std::filesystem::path>& paths  );
    // This function is called when window should close, if return value is true, window will stay open
    MRVIEWER_API bool interruptWindowClose();

    // Draw everything
    MRVIEWER_API void draw( bool force = false );
    // Draw 3d scene without UI
    MRVIEWER_API void drawScene() const;
    // Setup viewports views
    MRVIEWER_API void setupScene() const;
    // OpenGL context resize
    MRVIEWER_API void resize( int w, int h ); // explicitly set window size
    MRVIEWER_API void postResize( int w, int h ); // external resize due to user interaction
    MRVIEWER_API void postSetPosition( int xPos, int yPos ); // external set position due to user interaction
    MRVIEWER_API void postSetMaximized( bool maximized ); // external set maximized due to user interaction
    MRVIEWER_API void postSetIconified( bool iconified ); // external set iconified due to user interaction
    MRVIEWER_API void postFocus( bool focused ); // external focus handler due to user interaction
    MRVIEWER_API void postRescale( float x, float y ); // external rescale due to user interaction

    ////////////////////////
    // Multi-mesh methods //
    ////////////////////////

    // reset objectRoot with newRoot, append all RenderObjects and basis objects
    MRVIEWER_API void set_root( Object& newRoot );

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
    MRVIEWER_API ViewportId append_viewport( Viewport::ViewportRectangle viewportRect, bool append_empty = false );

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
    MRVIEWER_API void preciseFitDataViewport( MR::ViewportMask vpList = MR::ViewportMask::all(),
                                              const Viewport::FitDataParams& params = Viewport::FitDataParams() );

    size_t getTotalFrames() const { return frameCounter_.totalFrameCounter; }
    size_t getSwappedFrames() const { return frameCounter_.swappedFrameCounter; }
    size_t getFPS() const { return frameCounter_.fps; }
    long long getPrevFrameDrawTimeMillisec() const { return frameCounter_.drawTimeMilliSec; }

    // Returns memory amount used by shared GL memory buffer
    MRVIEWER_API size_t getStaticGLBufferSize() const;

    // Sets minimum auto increment for force redraw frames after basic events (the smallest value is 2)
    MRVIEWER_API void setMinimumForceRedrawFramesAfterEvents( int minimumIncrement );
    // if true only last frame of force redraw after events will be swapped, otherwise each will be swapped
    bool swapOnLastPostEventsRedraw{ true };

    // Increment number of forced frames to redraw in event loop
    // if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not
    MRVIEWER_API void incrementForceRedrawFrames( int i = 1, bool swapOnLastOnly = false );

    // Returns true if current frame will be swapped on display
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
     * Captures part of window (redraw 3d scene over UI (without redrawing UI))
     * @param pos left-bottom corner of capturing area relative of left-down corner of window. default = size(0, 0)
     * @param size size of capturing area. default = size(0, 0) - auto size to right-top corner of window.
     */
    MRVIEWER_API Image captureScreenShot( const Vector2i& pos = Vector2i(), const Vector2i& size = Vector2i() );

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

    // Sets manager of viewer settings which loads user personal settings on beginning of app 
    // and saves it in app's ending
    MRVIEWER_API void setViewportSettingsManager( std::unique_ptr<IViewerSettingsManager> mng );
    MRVIEWER_API const std::unique_ptr<IViewerSettingsManager>& getViewportSettingsManager() const { return settingsMng_; }

    struct PointInAllSpaces
    {
        // Screen space: window x[0, window.width] y[0, window.height]. [0,0] top-left corner of the window
        // Z [0,1] - 0 is Dnear, 1 is Dfar
        Vector3f screenSpace;

        // Viewport space: viewport x[0, viewport.width] y[0, viewport.height]. [0,0] top-left corner of the viewport with that Id
        // Z [0,1] - 0 is Dnear, 1 is Dfar
        Vector3f viewportSpace;
        ViewportId viewportId;

        // Clip space: viewport xyz[-1.f, 1.f]. [0, 0, -1] middle point of the viewport on Dnear. [0, 0, 1] middle point of the viewport on Dfar.
        // [-1, -1, -1] is lower left Dnear corner; [1, 1, 1] is upper right Dfar corner
        Vector3f clipSpace;

        // Camera space: applied view affine transform to world points xyz[-inf, inf]. [0, 0, 0] middle point of the viewport on Dnear.
        // X axis goes on the right. Y axis goes up. Z axis goes backward.
        Vector3f cameraSpace;

        // World space: applied model transform to Mesh(Point Cloud) vertices xyz[-inf, inf].
        Vector3f worldSpace;

        // Model space: coordinates as they stored in the model of VisualObject
        std::shared_ptr<VisualObject> obj;
        PointOnFace pof;
    };
    // Finds point in all spaces from screen space pixel point
    MRVIEWER_API PointInAllSpaces getPixelPointInfo( const Vector3f& screenPoint ) const;
    // Finds point under mouse in all spaces and under mouse viewport id
    MRVIEWER_API PointInAllSpaces getMousePointInfo() const;

    // Converts screen space coordinate to viewport space coordinate
    // (0,0) if viewport does not exist
    // screen space: X [0,window_width], Y [0,window_height] - (0,0) is upper left of window
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0,1] - 0 is Dnear, 1 is Dfar
    MRVIEWER_API Vector3f screenToViewport( const Vector3f& screenPoint, ViewportId id ) const;
    // Converts viewport space coordinate to screen space coordinate
    // (0,0) if viewport does not exist
    // screen space: X [0,window_width], Y [0,window_height] - (0,0) is upper left of window
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

    // This method is called after successful scene saving to update scene root, window title and undo
    MRVIEWER_API void onSceneSaved( const std::filesystem::path& savePath );

    // Get/Set menu plugin (which is separated from other plugins to be inited first before splash window starts)
    MRVIEWER_API const std::shared_ptr<ImGuiMenu>& getMenuPlugin() const;
    MRVIEWER_API void setMenuPlugin( std::shared_ptr<ImGuiMenu> menu );

    // Get the menu plugin casted in given type
    template <typename T>
    std::shared_ptr<T> getMenuPluginAs() const { return std::dynamic_pointer_cast<T>( getMenuPlugin() ); }

    // sets stop event loop flag (this flag is glfwShouldWindowClose equivalent)
    void stopEventLoop() { stopEventLoop_ = true; }
    // get stop event loop flag (this flag is glfwShouldWindowClose equivalent)
    bool getStopEventLoopFlag() const { return stopEventLoop_; }

    // return true if window should close
    // calls interrupt signal and if no slot interrupts return true, otherwise return false
    bool windowShouldClose();

    // returns true if viewer has valid GL context
    // note that sometimes it is not enough, for example to free GL memory in destructor,
    // glInitialized_ can be already reset and it requires `loadGL()` check too
    bool isGLInitialized() const { return glInitialized_; }

    // returns true if developer features are enabled, false otherwise
    bool isDeveloperFeaturesEnabled() const { return enableDeveloperFeatures_; }

    // update the title of the main window and, if any scene was opened, show its filename
    MRVIEWER_API void makeTitleFromSceneRootPath();

public:
    //////////////////////
    // Member variables //
    //////////////////////
    GLFWwindow* window;

    // Stores all the viewing options
    std::vector<Viewport> viewport_list;
    size_t selected_viewport_index;

    // List of registered plugins
    std::vector<ViewerPlugin*> plugins;

    // Store of recently opened files
    RecentFilesStore recentFilesStore;

    MouseController mouseController;

    int window_width; // current width
    int window_height; // current height
    Vector2i windowSavePos; // pos to save
    Vector2i windowSaveSize; // size to save
    Vector2i windowOldPos;
    bool windowMaximized{ false };

    // Stores basis axes meshes
    bool defaultLabelsBasisAxes{ false };
    bool defaultLabelsGlobalBasisAxes{ false };
    // if true - calls render without system events
    bool isAnimating{ false };
    // max fps if animating
    int animationMaxFps{ 30 };
    // this parameter can force up/down mouse scroll
    // useful for WebAssembler version because it has too powerful scroll
    float scrollForce{ 1.0f };

    std::unique_ptr<ObjectMesh> basisAxes;
    std::unique_ptr<ObjectMesh> globalBasisAxes;
    std::unique_ptr<ObjectMesh> rotationSphere;
    // Stores clipping plane mesh
    std::unique_ptr<ObjectMesh> clippingPlaneObject;

    // the window title that should be always displayed
    std::string defaultWindowTitle;

    //*********
    // SIGNALS
    //*********
    struct SignalStopHandler
    {
        using result_type = bool;

        template<typename Iter>
        bool operator()( Iter first, Iter last ) const
        {
            while ( first != last )
            {
                if ( *first )
                    return true; // slots execution stops if one returns true
                ++first;
            }
            return false;
        }
    };
    // Mouse events
    using MouseUpDownSignal = boost::signals2::signal<bool( MouseButton btn, int modifier ), SignalStopHandler>;
    using MouseMoveSignal = boost::signals2::signal<bool( int x, int y ), SignalStopHandler>;
    using MouseScrollSignal = boost::signals2::signal<bool( float delta ), SignalStopHandler>;
    MouseUpDownSignal mouseDownSignal; // signal is called on mouse down
    MouseUpDownSignal mouseUpSignal; // signal is called on mouse up
    MouseMoveSignal mouseMoveSignal; // signal is called on mouse move, note that input x and y are in screen space
    MouseScrollSignal mouseScrollSignal; // signal is called on mouse is scrolled
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
    // Render events
    using RenderSignal = boost::signals2::signal<void()>;
    RenderSignal preDrawSignal; // signal is called before scene draw (but after scene setup)
    RenderSignal preDrawPostViewportSignal; // signal is called before scene draw but after viewport.preDraw()
    RenderSignal drawSignal; // signal is called on scene draw (after objects tree but before viewport.postDraw())
    RenderSignal postDrawPreViewportSignal; // signal is called after scene draw but after before viewport.postDraw()
    RenderSignal postDrawSignal; // signal is called after scene draw
    // Scene events
    using DragDropSignal = boost::signals2::signal<bool( const std::vector<std::filesystem::path>& paths ), SignalStopHandler>;
    using PostResizeSignal = boost::signals2::signal<void( int x, int y )>;
    using PostRescaleSignal = boost::signals2::signal<void( float xscale, float yscale )>;
    using InterruptCloseSignal = boost::signals2::signal<bool(), SignalStopHandler>;
    DragDropSignal dragDropSignal; // signal is called on drag and drop file
    PostResizeSignal postResizeSignal; // signal is called after window resize
    PostRescaleSignal postRescaleSignal; // signal is called after window rescale
    InterruptCloseSignal interruptCloseSignal; // signal is called before close window (return true will prevent closing)

    // queue to ignore multiple mouse moves in one frame
    struct MouseQueueEvent
    {
        enum Type
        {
            Down,
            Up,
            Move,
            Drop // drop is here to know exact drop position 
            // Scroll is doubtable
        } type;
        std::function<void()> callEvent;
        // Constructor for emplace (for clang support)
        MouseQueueEvent( Type typeP, std::function<void()> callEventP ) :
            type{ typeP },
            callEvent{ callEventP }
        {}
    };
    std::queue<MouseQueueEvent> mouseEventQueue;
private:
    Viewer();
    ~Viewer();

    // Init window
    int launchInit_( const LaunchParams& params );
    // Return true if OpenGL loaded successfully
    bool checkOpenGL_(const LaunchParams& params );
    // Init base objects
    void init_();
    // Init all plugins on start
    void initPlugins_();
    // Shut all plugins at the end
    void shutdownPlugins_();
    // Search for python script to run or file to open on init
    void parseCommandLine_( int argc, char** argv );

    // process all events stored in mouseEventsQueue
    void processMouseEventsQueue_();

#ifdef __EMSCRIPTEN__
    static void mainLoopFunc_();
#endif

    // minimum auto increment force redraw frames after events
    int forceRedrawMinimumIncrement_{ 4 };
    // the minimum number of frames to be rendered even if the scene is unchanged
    int forceRedrawFrames_{ 0 };
    // Should be `<= forceRedrawFrames_`. The next N frames will not be shown on screen.
    int forceRedrawFramesWithoutSwap_{ 0 };

    // if this flag is set shows some developer features useful for debugging
    bool enableDeveloperFeatures_{ false };

    // special plugin for menu (initialized before splash window starts)
    std::shared_ptr<ImGuiMenu> menuPlugin_;

    mutable struct FrameCounter
    {
        size_t totalFrameCounter{ 0 };
        size_t swappedFrameCounter{ 0 };
        size_t startFrameNum{ 0 };
        size_t fps{ 0 };
        long long drawTimeMilliSec{ 0 };
        void startDraw();
        void endDraw( bool swapped );
        void reset();
    private:
        long long startFPSTime_{ 0 };
        long long startDrawTime_{ 0 };
    } frameCounter_;

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
    void resetRedraw_() const;

    enum class VisualObjectRenderType
    {
        Opaque,
        Transparent,
        NoDepthTest
    };

    VisualObjectRenderType getObjRenderType_( const VisualObject* obj, ViewportId viewportId ) const;

    void recursiveDraw_( const Viewport& vp, const Object& obj, const AffineXf3f& parentXf, VisualObjectRenderType renderType, int* numDraws = nullptr ) const;

    void initGlobalBasisAxesObject_();
    void initBasisAxesObject_();
    void initClippingPlaneObject_();
    void initRotationCenterObject_();

    bool stopEventLoop_{ false };

    bool isLaunched_{ false };
    // this flag is needed to know if all viewer setup was already done, and we can call draw
    bool focusRedrawReady_{ false };

    std::unique_ptr<AlphaSortGL> alphaSorter_;

    bool alphaSortEnabled_{false};

    bool glInitialized_{ false };

    bool isInDraw_{ false };

    ViewportId getFirstAvailableViewportId_() const;
    ViewportMask presentViewportsMask_;

    std::unique_ptr<IViewerSettingsManager> settingsMng_;

    std::shared_ptr<HistoryStore> globalHistoryStore_;

    std::shared_ptr<SpaceMouseHandler> spaceMouseHandler_;

    friend MRVIEWER_API Viewer& getViewerInstance();
};

// starts default viewer with given params and setup
MRVIEWER_API int launchDefaultViewer( const Viewer::LaunchParams& params, const ViewerSetup& setup );

// call this function to load MRViewer.dll
MRVIEWER_API void loadMRViewerDll();

} // end namespace