#include "MRViewer.h"
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRArrow.h>
#include <MRMesh/MRHistoryStore.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MRToFromEigen.h>
#include <MRMesh/MRTimer.h>
#include "MRMesh/MRUVSphere.h"
#include "MRMesh/MREmbeddedPython.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRDistanceMapLoad.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRAlphaSortGL.h"
#include "MRGLMacro.h"
#include "MRSetupViewer.h"
#include "MRGLStaticHolder.h"
#include "MRViewerPlugin.h"
#include "MRCommandLoop.h"
#include "MRSplashWindow.h"
#include "MRViewerSettingsManager.h"
#include "MRGladGlfw.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRPch/MRWasm.h"
#include "MRGetSystemInfoJson.h"
#include "MRSpaceMouseHandler.h"
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRViewer/MRRenderGLHelpers.h"

#ifndef __EMSCRIPTEN__
#include <boost/exception/diagnostic_information.hpp>
#include <boost/stacktrace.hpp>
#endif
#include "MRViewerIO.h"
#include "MRProgressBar.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRAppendHistory.h"
#include "MRSwapRootAction.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/html5.h>
#include "MRMesh/MRConfig.h"
#define GLFW_INCLUDE_ES3

namespace
{
    double sEmsPixelRatio = 1.0f;
}

extern "C"
{
EMSCRIPTEN_KEEPALIVE int resizeEmsCanvas( float width, float height )
{
    auto pixelRatio = emscripten_get_device_pixel_ratio();
    if ( sEmsPixelRatio != pixelRatio )
    {
        sEmsPixelRatio = pixelRatio;
        MR::getViewerInstance().postRescale( float( sEmsPixelRatio ), float( sEmsPixelRatio ) );
    }
    float newWidth = width * pixelRatio;
    float newHeight = height * pixelRatio;
    glfwSetWindowSize( MR::getViewerInstance().window, int( newWidth ), int( newHeight ) );
    MR::getViewerInstance().incrementForceRedrawFrames( MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents, false );
    return 1;
}

EMSCRIPTEN_KEEPALIVE void emsPostEmptyEvent( int forceFrames )
{
    auto& viewer = MR::getViewerInstance();
    viewer.incrementForceRedrawFrames( forceFrames, true );
    viewer.postEmptyEvent();
}

EMSCRIPTEN_KEEPALIVE void emsUpdateViewportBounds()
{
    auto& viewer = MR::getViewerInstance();
    auto bounds = viewer.getViewportsBounds();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( updateVPBounds($0,$1,$2,$3), bounds.min.x,bounds.min.y, MR::width( bounds ),MR::height( bounds ) );
#pragma clang diagnostic pop
}

EMSCRIPTEN_KEEPALIVE void emsForceSettingsSave()
{
    auto& viewer = MR::getViewerInstance();
    auto& settingsManager = viewer.getViewportSettingsManager();
    if ( settingsManager )
        settingsManager->saveSettings( viewer );
    MR::Config::instance().writeToFile();
}

}
#endif

static void glfw_mouse_press( GLFWwindow* /*window*/, int button, int action, int modifier )
{
    MR::Viewer::MouseButton mb;

    if ( button == GLFW_MOUSE_BUTTON_1 )
        mb = MR::Viewer::MouseButton::Left;
    else if ( button == GLFW_MOUSE_BUTTON_2 )
        mb = MR::Viewer::MouseButton::Right;
    else //if (button == GLFW_MOUSE_BUTTON_3)
        mb = MR::Viewer::MouseButton::Middle;

    auto* viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( {"Mouse press", [mb, action, modifier, viewer] ()
    {
        if ( action == GLFW_PRESS )
            viewer->mouseDown( mb, modifier );
        else
            viewer->mouseUp( mb, modifier );
    } } );
}

static void glfw_error_callback( int /*error*/, const char* description )
{
    spdlog::error( "glfw_error_callback: {}", description );
}

static void glfw_char_mods_callback( GLFWwindow* /*window*/, unsigned int codepoint )
{
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( { "Char", [codepoint, viewer] ()
    {
        viewer->keyPressed( codepoint, 0 );
    } } );
}

static void glfw_key_callback( GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int modifier )
{
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( {"Key press", [action, key, modifier, viewer] ()
    {
        if ( action == GLFW_PRESS )
            viewer->keyDown( key, modifier );
        else if ( action == GLFW_RELEASE )
            viewer->keyUp( key, modifier );
        else if ( action == GLFW_REPEAT )
            viewer->keyRepeat( key, modifier );
    } } );
}

static void glfw_framebuffer_size( GLFWwindow* /*window*/, int width, int height )
{
    auto viewer = &MR::getViewerInstance();
    viewer->postResize( width, height );
    viewer->postEmptyEvent();
}

static void glfw_window_pos( GLFWwindow* /*window*/, int xPos, int yPos )
{
    // need for remember window pos and size before maximize
    // located here because glfw_window_pos calls before glfw_window_maximize and glfw_window_iconify (experience)
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( { "Windows pos", [xPos, yPos, viewer] ()
    {
        viewer->windowOldPos = viewer->windowSavePos;
        viewer->postSetPosition( xPos, yPos );
    } } );
}

static void glfw_cursor_enter_callback( GLFWwindow* /*window*/, int entered )
{
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( { "Cursor enter", [entered, viewer] ()
    {
        viewer->cursorEntranceSignal( bool( entered ) );
    } } );
}

#ifndef __EMSCRIPTEN__
static void glfw_window_maximize( GLFWwindow* /*window*/, int maximized )
{
    MR::getViewerInstance().postSetMaximized( bool( maximized ) );
}

static void glfw_window_iconify( GLFWwindow* /*window*/, int iconified )
{
    MR::getViewerInstance().postSetIconified( bool( iconified ) );
}

static void glfw_window_focus( GLFWwindow* /*window*/, int focused )
{
    MR::getViewerInstance().postFocus( bool( focused ) );
}
#endif

static void glfw_window_scale( GLFWwindow* /*window*/, float xscale, float yscale )
{
    auto viewer = &MR::getViewerInstance();
    viewer->postRescale( xscale, yscale );
}

#if defined(__EMSCRIPTEN__) && defined(MR_EMSCRIPTEN_ASYNCIFY)
static constexpr int minEmsSleep = 3; // ms - more then 300 fps possible
static EM_BOOL emsEmptyCallback( double, void* )
{
    return EM_TRUE;
}
#endif

static void glfw_mouse_move( GLFWwindow* /*window*/, double x, double y )
{
    auto* viewer = &MR::getViewerInstance();
    auto eventCall = [x, y,viewer] ()
    {
        // scale mouse pos for retina to easier support different framebuffer and window size (glfw cursor pos is in window coords)
        viewer->mouseMove( int( std::round( x * viewer->pixelRatio ) ), int( std::round( y * viewer->pixelRatio ) ) );
        viewer->draw();
    };
    viewer->eventQueue.emplace( { "Mouse move", eventCall }, true );
}

static void glfw_mouse_scroll( GLFWwindow* /*window*/, double /*x*/, double y )
{
    static double prevY = 0.0;
    auto viewer = &MR::getViewerInstance();
    if ( prevY * y < 0.0 )
        viewer->eventQueue.popByName( "Mouse scroll" );
    auto eventCall = [y, viewer, prevPtr = &prevY] ()
    {
        *prevPtr = y;
        viewer->mouseScroll( float( y ) );
        viewer->draw();
    };
    viewer->eventQueue.emplace( { "Mouse scroll", eventCall } );
}

static void glfw_drop_callback( [[maybe_unused]] GLFWwindow *window, int count, const char **filenames )
{
    if ( count == 0 )
        return;

    std::vector<std::filesystem::path> paths( count );
    for ( int i = 0; i < count; ++i )
    {
        paths[i] = MR::pathFromUtf8( filenames[i] );
    }
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( { "Drop", [paths, viewer] ()
    {
        viewer->dragDrop( paths );
    } } );
    viewer->postEmptyEvent();
}

static void glfw_joystick_callback( int jid, int event )
{
    auto viewer = &MR::getViewerInstance();
    viewer->eventQueue.emplace( { "Joystick", [jid, event, viewer] ()
    {
        viewer->joystickUpdateConnected( jid, event );
    } } );
}

namespace MR
{

void addLabel( ObjectMesh& obj, const std::string& str, const Vector3f& pos )
{
    auto label = std::make_shared<ObjectLabel>();
    label->setFrontColor( Color::white(), false );
    label->setLabel( { str, pos } );
    label->setPivotPoint( Vector2f( 0.5f, 0.5f ) );
    label->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
    obj.addChild( label );
}

int launchDefaultViewer( const Viewer::LaunchParams& params, const ViewerSetup& setup )
{
    static bool firstLaunch = true;
    if ( !firstLaunch )
    {
        spdlog::error( "Viewer can be launched only once" );
        return 1;
    }
    else
    {
        firstLaunch = false;
    }

    auto& viewer = MR::Viewer::instanceRef();

    MR::setupLoggerByDefault();

    setup.setupBasePlugins( &viewer );
    setup.setupCommonModifiers( &viewer );
    setup.setupCommonPlugins( &viewer );
    setup.setupSettingsManager( &viewer, params.name );
    setup.setupConfiguration( &viewer );
    CommandLoop::appendCommand( [&] ()
    {
        setup.setupExtendedLibraries();
    }, CommandLoop::StartPosition::AfterSplashAppear );
#if defined(__EMSCRIPTEN__) || !defined(NDEBUG)
    return viewer.launch( params );
#else
    int res = 0;
    try
    {
        res = viewer.launch( params );
    }
    catch ( ... )
    {
        spdlog::critical( boost::current_exception_diagnostic_information() );
        spdlog::critical( "Exception stacktrace:\n{}", to_string( boost::stacktrace::stacktrace() ) );
        printCurrentTimerBranch();
        res = 1;
    }

    return res;
#endif
}

void loadMRViewerDll()
{
}

void Viewer::parseLaunchParams( LaunchParams& params )
{
    bool nextW{ false };
    bool nextH{ false };
    for ( int i = 1; i < params.argc; ++i )
    {
        std::string flag( params.argv[i] );
        if ( nextW )
        {
            nextW = false;
            params.width = std::atoi( flag.c_str() );
        }
        else if ( nextH )
        {
            nextH = false;
            params.height = std::atoi( flag.c_str() );
        }
        else if ( flag == "-noWindow" )
        {
            params.windowMode = LaunchParams::NoWindow;
            params.isAnimating = true;
        }
        else if ( flag == "-fullscreen" )
            params.fullscreen = true;
        else if ( flag == "-noClose" )
            params.close = false;
        else if ( flag == "-noEventLoop" )
            params.startEventLoop = false;
        else if ( flag == "-hidden" )
            params.windowMode = LaunchParams::Hide;
        else if ( flag == "-tryHidden" )
            params.windowMode = LaunchParams::TryHidden;
        else if ( flag == "-transparentBgOn" )
            params.enableTransparentBackground = true;
        else if ( flag == "-transparentBgOff" )
            params.enableTransparentBackground = false;
        else if ( flag == "-noSplash" )
            params.splashWindow.reset();
        else if ( flag == "-console" )
            params.console = true;
        else if ( flag == "-openGL3" )
            params.preferOpenGL3 = true;
        else if ( flag == "-noRenderInTexture" )
            params.render3dSceneInTexture = false;
        else if ( flag == "-develop" )
            params.developerFeatures = true;
        else if ( flag == "-width" )
            nextW = true;
        else if ( flag == "-height" )
            nextH = true;
    }
}

#ifdef __EMSCRIPTEN__
#ifndef MR_EMSCRIPTEN_ASYNCIFY
void Viewer::emsMainInfiniteLoop()
{
    auto& viewer = getViewerInstance();
    viewer.draw( true );
    viewer.eventQueue.execute();
    CommandLoop::processCommands();
}
#endif

void Viewer::mainLoopFunc_()
{
#ifdef MR_EMSCRIPTEN_ASYNCIFY
    for (;;)
    {
        if ( isAnimating )
        {
            const double minDuration = 1e3 / double( animationMaxFps );
            emscripten_sleep( std::max( int( minDuration ), minEmsSleep ) );
        }
        else if ( !isAnimating && eventQueue.empty() )
        {
            emscripten_sleep( minEmsSleep ); // more then 300 fps possible
            continue;
        }

        do
        {
            draw( true );
            eventQueue.execute();
            CommandLoop::processCommands();
        } while ( forceRedrawFrames_ > 0 || needRedraw_() );
    }
#else
    emscripten_set_main_loop( emsMainInfiniteLoop, 0, true );
#endif
}
#endif

int Viewer::launch( const LaunchParams& params )
{
    if ( isLaunched_ )
    {
        spdlog::error( "Viewer is already launched!" );
        return 1;
    }

    // log start line
    for ( int i = 0; i < params.argc; ++i )
        spdlog::info( "argv[{}]: {}", i, params.argv[i] );

    isAnimating = params.isAnimating;
    animationMaxFps = params.animationMaxFps;
    enableDeveloperFeatures_ = params.developerFeatures;
    auto res = launchInit_( params );
    if ( res != EXIT_SUCCESS )
        return res;

    CommandLoop::setState( CommandLoop::StartPosition::AfterSplashHide );
    CommandLoop::processCommands(); // execute pre init commands before first draw
    focusRedrawReady_ = true;

    if ( params.windowMode == LaunchParams::HideInit && window )
        glfwShowWindow( window );

    parseCommandLine_( params.argc, params.argv );

    CommandLoop::setState( CommandLoop::StartPosition::AfterWindowAppear );

    if ( params.startEventLoop )
    {
#ifdef __EMSCRIPTEN__
        mainLoopFunc_();
#else
        launchEventLoop();
#endif
    }
    if ( params.close )
        launchShut();
    return EXIT_SUCCESS;
}

bool Viewer::checkOpenGL_(const LaunchParams& params )
{
    int windowWidth = params.width;
    int windowHeight = params.height;
#ifdef __APPLE__
    alphaSorter_.reset();
    spdlog::warn( "Alpha sort is not available" );

    spdlog::warn( "Loading OpenGL 4.1 for macOS" );
    if ( !tryCreateWindow_( params.fullscreen, windowWidth, windowHeight, params.name, 4, 1 ) )
    {
        spdlog::critical( "Cannot load OpenGL 4.1" );
        return false;
    }
#else
#ifdef __EMSCRIPTEN__
    alphaSorter_.reset();
    spdlog::warn( "Alpha sort is not available" );
    spdlog::warn( "Loading WebGL 2 (OpenGL ES 3.0)" );
    if ( !tryCreateWindow_( params.fullscreen, windowWidth, windowHeight, params.name, 3, 3 ) )
    {
        spdlog::critical( "Cannot load WebGL 2 (OpenGL ES 3.0)" );
        return false;
    }
#else
    if ( params.preferOpenGL3 || !tryCreateWindow_( params.fullscreen, windowWidth, windowHeight, params.name, 4, 3 ) )
    {
        alphaSorter_.reset();
        if ( !params.preferOpenGL3 )
            spdlog::warn( "Cannot load OpenGL 4.3, try load OpenGL 3.3" );
        if ( !tryCreateWindow_( params.fullscreen, windowWidth, windowHeight, params.name, 3, 3 ) )
        {
            spdlog::critical( "Cannot load OpenGL 3.3" );
#ifdef _WIN32
            MessageBoxA( NULL, "Cannot activate OpenGL 3.3.\n"
                "Please verify that you have decent graphics card and its drivers are installed.",
                "MeshInspector/MeshLib Error", MB_OK );
#endif
            return false;
        }
        spdlog::warn( "Alpha sort is not available" );
    }
#endif
#endif
    return true;
}

int Viewer::launchInit_( const LaunchParams& params )
{
    CommandLoop::setMainThreadId( std::this_thread::get_id() );
    spdlog::info( "Log file: {}", utf8string( Logger::instance().getLogFileName() ) );
    glfwSetErrorCallback( glfw_error_callback );
    if ( !glfwInit() )
    {
        spdlog::error( "glfwInit failed" );
        return EXIT_FAILURE;
    }
    spdlog::info( "glfwInit succeeded" );
#if defined(__APPLE__)
    //Setting window properties
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint( GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE );
#endif
    if ( !settingsMng_ )
        glfwWindowHint( GLFW_SAMPLES, 8 );
    else
        glfwWindowHint( GLFW_SAMPLES, settingsMng_->loadInt( "multisampleAntiAliasing", 8 ) );
#ifndef __EMSCRIPTEN__
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_FOCUS_ON_SHOW, GLFW_TRUE );
    glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, params.enableTransparentBackground );
#endif

    alphaSorter_ = std::make_unique<AlphaSortGL>();
    if ( params.render3dSceneInTexture )
        sceneTexture_ = std::make_unique<SceneTextureGL>();

    glfwWindowHint( GLFW_VISIBLE, int( bool( params.windowMode == LaunchParams::Show ) ) );
    bool windowMode = params.windowMode != LaunchParams::NoWindow;

    if ( windowMode )
    {
        if ( !checkOpenGL_(params) )
        {
            if ( params.windowMode == LaunchParams::TryHidden )
                windowMode = false;
            else
                return EXIT_FAILURE;
        }
    }

    if ( windowMode )
    {
        assert( window );

        glfwMakeContextCurrent( window );
        if ( !loadGL() )
        {
            spdlog::error( "Failed to load OpenGL and its extensions" );
            return( -1 );
        }
        glInitialized_ = true;
#ifndef __EMSCRIPTEN__
        spdlog::info( "OpenGL Version {}.{} loaded", GLVersion.major, GLVersion.minor );
#endif
        int major, minor, rev;
        major = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MAJOR );
        minor = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MINOR );
        rev = glfwGetWindowAttrib( window, GLFW_CONTEXT_REVISION );
        spdlog::info( "OpenGL version received: {}.{}.{}", major, minor, rev );
        if ( glInitialized_ )
        {
            spdlog::info( "Supported OpenGL is {}", ( const char* )glGetString( GL_VERSION ) );
            spdlog::info( "Supported GLSL is {}", ( const char* )glGetString( GL_SHADING_LANGUAGE_VERSION ) );
        }
        defaultWindowTitle = params.name;
        if ( params.showMRVersionInTitle )
            defaultWindowTitle += " (" + GetMRVersionString() + ")";
        glfwSetWindowTitle( window, defaultWindowTitle.c_str() );

        glfwSetInputMode( window, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
        // Register callbacks
        glfwSetKeyCallback( window, glfw_key_callback );
        glfwSetCursorPosCallback( window, glfw_mouse_move );
        glfwSetFramebufferSizeCallback( window, glfw_framebuffer_size );
        glfwSetWindowPosCallback( window, glfw_window_pos );
        glfwSetCursorEnterCallback( window, glfw_cursor_enter_callback );
#ifndef __EMSCRIPTEN__
        glfwSetWindowMaximizeCallback( window, glfw_window_maximize );
        glfwSetWindowIconifyCallback( window, glfw_window_iconify );
        glfwSetWindowContentScaleCallback( window, glfw_window_scale );
        glfwSetWindowFocusCallback( window, glfw_window_focus );
#endif
        glfwSetMouseButtonCallback( window, glfw_mouse_press );
        glfwSetCharCallback( window, glfw_char_mods_callback );
        glfwSetDropCallback( window, glfw_drop_callback );
        glfwSetJoystickCallback( glfw_joystick_callback );
        // Handle retina displays (windows and mac)
        int width, height;
        glfwGetFramebufferSize( window, &width, &height );
        // Initialize IGL viewer
        glfw_framebuffer_size( window, width, height );

#ifdef __APPLE__
        int winWidth, winHeight;
        glfwGetWindowSize( window, &winWidth, &winHeight );
        pixelRatio = float( width ) / float( winWidth );
#endif

        float xscale{ 1.0f }, yscale{ 1.0f };
#ifndef __EMSCRIPTEN__
        glfwGetWindowContentScale( window, &xscale, &yscale );
#endif
        glfw_window_scale( window, xscale, yscale );

        enableAlphaSort( true );
        if ( sceneTexture_ )
            sceneTexture_->reset( { width, height }, -1 );

        if ( alphaSorter_ )
        {
            alphaSorter_->init();
            alphaSorter_->updateTransparencyTexturesSize( width, height );
        }

        mouseController.connect();
        touchesController.connect( this );
        spaceMouseController.connect();
        initSpaceMouseHandler_();
        touchpadController.connect( this );
        touchpadController.initialize( window );
    }

    CommandLoop::setState( CommandLoop::StartPosition::AfterWindowInit );
    CommandLoop::processCommands();

    std::future<void> splashMinTimer;
    if ( windowMode && params.windowMode != LaunchParams::Hide && params.splashWindow )
    {
        params.splashWindow->start();
        // minimum time splash screen to stay present
        splashMinTimer = std::async( std::launch::async, [seconds = params.splashWindow->minimumTimeSec()] ()
        {
            std::this_thread::sleep_for( std::chrono::duration<float>( seconds ) );
        } );
    }

    CommandLoop::setState( CommandLoop::StartPosition::AfterSplashAppear );
    CommandLoop::processCommands();

    if ( menuPlugin_ )
    {
        spdlog::info( "Init menu plugin." );
        menuPlugin_->init( this );
    }

    // print after menu init to know valid menu_scaling
    spdlog::info( "System info:\n{}", GetSystemInfoJson().toStyledString() );

    init_();
    // it is replaced here because some plugins can rise modal window, and scroll event sometimes can pass over it
    if ( window )
        glfwSetScrollCallback( window, glfw_mouse_scroll );
    // give it name of app to store in right place
    recentFilesStore = RecentFilesStore( params.name );

    CommandLoop::setState( CommandLoop::StartPosition::AfterPluginInit );
    CommandLoop::processCommands();

    if ( windowMode && params.windowMode != LaunchParams::Hide && params.splashWindow )
    {
        splashMinTimer.get();
        params.splashWindow->stop();
    }

    // important to be after splash
    if ( menuPlugin_ )
        menuPlugin_->initBackend();

    isLaunched_ = true;

    return EXIT_SUCCESS;
}

void Viewer::launchEventLoop()
{
    if ( !isLaunched_ )
    {
        spdlog::error( "Viewer is not launched!" );
        return;
    }

    // Rendering loop
    while ( !windowShouldClose() )
    {
        do
        {
            draw( true );
            glfwPollEvents();
            eventQueue.execute();
            if ( spaceMouseHandler_ )
                spaceMouseHandler_->handle();
            CommandLoop::processCommands();
        } while ( ( !( window && glfwWindowShouldClose( window ) ) && !stopEventLoop_ ) && ( forceRedrawFrames_ > 0 || needRedraw_() ) );

        if ( isAnimating )
        {
            const double minDuration = 1.0 / double( animationMaxFps );
            glfwWaitEventsTimeout( minDuration );
            eventQueue.execute();
        }
        else
        {
            glfwWaitEvents();
            eventQueue.execute();
        }
        if ( spaceMouseHandler_ )
            spaceMouseHandler_->handle();
    }
}

void Viewer::launchShut()
{
    if ( !isLaunched_ )
    {
        spdlog::error( "Viewer is not launched!" );
        return;
    }
    if ( window )
        glfwHideWindow( window );

    if ( settingsMng_ )
    {
        spdlog::info( "Save user settings." );
        settingsMng_->saveSettings( *this );
    }

    for ( auto& viewport : viewport_list )
        viewport.shut();
    shutdownPlugins_();

    // Clear plugins
    plugins.clear();

    // Clear objects
    SceneRoot::get().removeAllChildren();
    basisAxes.reset();
    rotationSphere.reset();
    clippingPlaneObject.reset();
    globalBasisAxes.reset();
    globalHistoryStore_.reset();

    GLStaticHolder::freeAllShaders();

    alphaSorter_.reset();
    sceneTexture_.reset();

    touchpadController.reset();

    glfwDestroyWindow( window );
    glfwTerminate();
    glInitialized_ = false;
    isLaunched_ = false;
    spaceMouseHandler_.reset();
}

void Viewer::init_()
{
    initBasisAxesObject_();
    initClippingPlaneObject_();
    initRotationCenterObject_();
    initGlobalBasisAxesObject_();

    initPlugins_();

    auto& mainViewport = viewport();
    if ( settingsMng_ )
    {
        spdlog::info( "Load user settings." );
        settingsMng_->loadSettings( *this );
    }
    mainViewport.init();
}

void Viewer::initPlugins_()
{
    // Init all plugins
    for ( unsigned int i = 0; i < plugins.size(); ++i )
    {
        plugins[i]->init( this );
    }
}

void Viewer::shutdownPlugins_()
{
    for ( unsigned int i = 0; i < plugins.size(); ++i )
    {
        plugins[i]->shutdown();
    }
    if ( menuPlugin_ )
        menuPlugin_->shutdown();
}

void Viewer::parseCommandLine_( [[maybe_unused]] int argc, [[maybe_unused]] char** argv )
{
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_PYTHON)
    std::vector<std::filesystem::path> supportedFiles;
    for ( int i = 1; i < argc; ++i )
    {
        const auto argAsPath = pathFromUtf8( argv[i] );
        if( EmbeddedPython::isPythonScript( argAsPath ) )
        {
            EmbeddedPython::init();
            // Draw twice to show all menus on screen
            {
                draw( true );
                draw( true );
            }
            EmbeddedPython::setupArgv( argc - i, &argv[i] );
            EmbeddedPython::runScript( argAsPath );
            // Draw to update after executing script
            {
                draw( true );
            }
            EmbeddedPython::finalize();
            break;
        }
        if ( isSupportedFormat( argAsPath ) )
            supportedFiles.push_back( argAsPath );
    }
    loadFiles( supportedFiles );
#endif
}

void Viewer::EventQueue::emplace( NamedEvent event, bool skipable )
{
    std::unique_lock lock( mutex_ );
    if ( queue_.empty() || !skipable || !lastSkipable_ )
        queue_.emplace( std::move( event ) );
    else
        queue_.back() = std::move( event );
    lastSkipable_ = skipable;
}

void Viewer::EventQueue::execute()
{
    std::unique_lock lock( mutex_ );
    while ( !queue_.empty() )
    {
        if ( queue_.front().cb )
            queue_.front().cb();
        queue_.pop();
    }
}

bool Viewer::EventQueue::empty() const
{
    std::unique_lock lock( mutex_ );
    return queue_.empty();
}

void Viewer::EventQueue::popByName( const std::string& name )
{
    std::unique_lock lock( mutex_ );
    while ( !queue_.empty() && queue_.front().name == name )
        queue_.pop();
}

void Viewer::postEmptyEvent()
{
    if ( !isGLInitialized() )
        return;
#ifdef __EMSCRIPTEN__
    eventQueue.emplace( { "Empty", [] () {} } );
#endif
    glfwPostEmptyEvent();
}

Viewer::Viewer() :
    selected_viewport_index( 0 )
{
    window = nullptr;

    viewport_list.reserve( 32 );
    viewport_list.emplace_back();
    viewport_list.front().id = ViewportId{ 1 };
    presentViewportsMask_ |= viewport_list.front().id;
}

Viewer::~Viewer()
{
    glInitialized_ = false;
    alphaSorter_.reset();
    sceneTexture_.reset();
}

bool Viewer::isSupportedFormat( const std::filesystem::path& mesh_file_name )
{
    std::error_code ec;
    if( !std::filesystem::exists( mesh_file_name, ec ) )
        return false;
    if( !std::filesystem::is_regular_file( mesh_file_name, ec ) )
        return false;

    std::string ext = utf8string( mesh_file_name.extension() );
    for( auto& c : ext )
        c = (char) tolower( c );

    for( auto& filter : MeshLoad::getFilters() )
    {
        if( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : LinesLoad::Filters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : PointsLoad::Filters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_DICOM ) && !defined(MRMESH_NO_VOXEL)
    for ( auto& filter : VoxelsLoad::Filters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
#endif
    for ( auto& filter : DistanceMapLoad::Filters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : SceneFileFilters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }

    return false;
}

bool Viewer::loadFiles( const std::vector< std::filesystem::path>& filesList )
{
    if ( filesList.empty() )
        return false;

    ProgressBar::orderWithMainThreadPostProcessing( "Open files", [filesList] ()->std::function<void()>
    {
        std::vector<std::filesystem::path> loadedFiles;
        std::vector<std::string> errorList;
        std::string loadInfoRes;
        std::vector<std::shared_ptr<Object>> loadedObjects;
        for ( int i = 0; i < filesList.size(); ++i )
        {
            const auto& filename = filesList[i];
            if ( filename.empty() )
                continue;

            spdlog::info( "Loading file {}", utf8string( filename ) );
            std::string loadInfo;
            auto res = loadObjectFromFile( filename, &loadInfo, [callback = ProgressBar::callBackSetProgress, i, number = filesList.size()]( float v )
            {
                return callback( ( i + v ) / number );
            } );
            spdlog::info( "Load file {} - {}", utf8string( filename ), res.has_value() ? "success" : res.error().c_str() );
            if ( !res.has_value() )
            {
                errorList.push_back( std::move( res.error() ) );
                continue;
            }

            if ( !loadInfo.empty() )
            {
                loadInfoRes += ( ( loadInfoRes.empty() ? "" : "\n" ) + utf8string( filename ) + ":\n" + loadInfo + "\n" );
            }


            auto& newObjs = *res;
            bool anyObjLoaded = false;
            for ( auto& obj : newObjs )
            {
                if ( !obj )
                    continue;

                anyObjLoaded = true;
                loadedObjects.push_back( obj );
            }
            if ( anyObjLoaded )
                loadedFiles.push_back( filename );
            else
                errorList.push_back( "No objects found in the file \"" + utf8string( filename ) + "\"" );
        }
        return [loadedObjects, loadedFiles, errorList, loadInfoRes]
        {
            if ( !loadedObjects.empty() )
            {
                bool sceneFile = std::string( loadedObjects[0]->typeName() ) == std::string( Object::TypeName() );
                bool sceneEmpty = SceneRoot::get().children().empty();
                if ( loadedObjects.size() == 1 && sceneFile )
                {
                    AppendHistory<SwapRootAction>( "Load Scene File" );
                    auto newRoot = loadedObjects[0];
                    std::swap( newRoot, SceneRoot::getSharedPtr() );
                }
                else
                {
                    std::string historyName = loadedObjects.size() == 1 ? "Open file" : "Open files";
                    SCOPED_HISTORY( historyName );
                    for ( auto& obj : loadedObjects )
                    {
                        AppendHistory<ChangeSceneAction>( "Load File", obj, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( obj );
                    }
                    auto& viewerInst = getViewerInstance();
                    for ( const auto& file : loadedFiles )
                        viewerInst.recentFilesStore.storeFile( file );
                }
                if ( loadedFiles.size() == 1 && ( sceneFile || sceneEmpty ) )
                {
                    auto path = loadedFiles[0];
                    if ( !sceneFile )
                        path.replace_extension( ".mru" );
                    getViewerInstance().onSceneSaved( path, sceneFile );
                }
                getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f } );
            }
            if ( !errorList.empty() )
            {
                std::string errorAll;
                for ( auto& error : errorList )
                    errorAll += "\n" + error;
                showError( errorAll.substr( 1 ) );
            }
            else if ( !loadInfoRes.empty() )
            {
                showModal( loadInfoRes, ImGuiMenu::ModalMessageType::Warning );
            }
        };
    } );

    return true;
}

bool Viewer::saveToFile( const std::filesystem::path & path )
{
    auto obj = getDepthFirstObject<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    auto res = saveObjectToFile( *obj, path );
    if ( !res.has_value() )
        return false;
    return true;
}

bool Viewer::keyPressed( unsigned int unicode_key, int modifiers )
{
    // repeated signals swap each frame to prevent freezes
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, false );

    eventsCounter_.counter[size_t( EventType::CharPressed )]++;

    return charPressedSignal( unicode_key, modifiers );
}

bool Viewer::keyDown( int key, int modifiers )
{
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::KeyDown )]++;

    if ( keyDownSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::keyUp( int key, int modifiers )
{
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::KeyUp )]++;

    if ( keyUpSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::keyRepeat( int key, int modifiers )
{
    // repeated signals swap each frame to prevent freezes
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, false );

    eventsCounter_.counter[size_t( EventType::KeyRepeat )]++;

    if ( keyRepeatSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::mouseDown( MouseButton button, int modifier )
{
    // if the mouse was released in this frame, then we need to render at least one more frame to get button reaction;
    // if the mouse was pressed and released in this frame, then at least two more frames are necessary because of
    // g_MouseJustPressed in ImGui_ImplGlfw_UpdateMousePosAndButtons
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::MouseDown )]++;

    if ( mouseDownSignal( button, modifier ) )
        return true;

    return true;
}

bool Viewer::mouseUp( MouseButton button, int modifier )
{
    // if the mouse was released in this frame, then we need to render at least one more frame to get button reaction;
    // if the mouse was pressed and released in this frame, then at least two more frames are necessary because of
    // g_MouseJustPressed in ImGui_ImplGlfw_UpdateMousePosAndButtons
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::MouseUp )]++;

    if ( mouseUpSignal( button, modifier ) )
        return true;

    return true;
}

bool Viewer::mouseMove( int mouse_x, int mouse_y )
{
    eventsCounter_.counter[size_t( EventType::MouseMove )]++;

    if ( mouseMoveSignal( mouse_x, mouse_y ) )
        return true;

    return false;
}

bool Viewer::touchStart( int id, int x, int y )
{
    return touchStartSignal( id, x, y );
}

bool Viewer::touchMove( int id, int x, int y )
{
    return touchMoveSignal( id, x, y );
}

bool Viewer::touchEnd( int id, int x, int y )
{
    return touchEndSignal( id, x, y );
}

bool Viewer::touchpadRotateGestureBegin()
{
    return touchpadRotateGestureBeginSignal();
}

bool Viewer::touchpadRotateGestureUpdate( float angle )
{
    return touchpadRotateGestureUpdateSignal( angle );
}

bool Viewer::touchpadRotateGestureEnd()
{
    return touchpadRotateGestureEndSignal();
}

bool Viewer::touchpadSwipeGestureBegin()
{
    return touchpadSwipeGestureBeginSignal();
}

bool Viewer::touchpadSwipeGestureUpdate( float dx, float dy, bool kinetic )
{
    return touchpadSwipeGestureUpdateSignal( dx, dy, kinetic );
}

bool Viewer::touchpadSwipeGestureEnd()
{
    return touchpadSwipeGestureEndSignal();
}

bool Viewer::touchpadZoomGestureBegin()
{
    return touchpadZoomGestureBeginSignal();
}

bool Viewer::touchpadZoomGestureUpdate( float scale, bool kinetic )
{
    return touchpadZoomGestureUpdateSignal( scale, kinetic );
}

bool Viewer::touchpadZoomGestureEnd()
{
    return touchpadZoomGestureEndSignal();
}

bool Viewer::mouseScroll( float delta_y )
{
    eventsCounter_.counter[size_t( EventType::MouseScroll )]++;

    if ( mouseScrollSignal( scrollForce * delta_y ) )
        return true;

    return true;
}

bool Viewer::spaceMouseMove( const Vector3f& translate, const Vector3f& rotate )
{
    return spaceMouseMoveSignal( translate, rotate );
}

bool Viewer::spaceMouseDown( int key )
{
    return spaceMouseDownSignal( key );
}

bool Viewer::spaceMouseUp( int key )
{
    return spaceMouseUpSignal( key );
}

bool Viewer::spaceMouseRepeat( int key )
{
    return spaceMouseRepeatSignal( key );
}

bool Viewer::dragDrop( const std::vector<std::filesystem::path>& paths )
{
    if ( dragDropSignal( paths ) )
        return true;

    return false;
}

bool Viewer::interruptWindowClose()
{
    if ( interruptCloseSignal() )
        return true;

    return false;
}

void Viewer::joystickUpdateConnected( int jid, int event )
{
    if ( spaceMouseHandler_ )
        spaceMouseHandler_->updateConnected( jid, event );
}

static bool getRedrawFlagRecursive( const Object& obj, ViewportMask mask )
{
    if ( obj.getRedrawFlag( mask ) )
        return true;
    if ( !obj.isVisible( mask ) )
        return false;
    for ( const auto& child : obj.children() )
    {
        if ( getRedrawFlagRecursive( *child, mask ) )
            return true;
    }
    return false;
}

static void resetRedrawFlagRecursive( const Object& obj )
{
    obj.resetRedrawFlag();
    for ( const auto& child : obj.children() )
        resetRedrawFlagRecursive( *child );
}

bool Viewer::tryCreateWindow_( bool fullscreen, int& width, int& height, const std::string& name, int major, int minor )
{
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, major );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, minor );
    if ( fullscreen )
    {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode( monitor );
        window = glfwCreateWindow( mode->width, mode->height, name.c_str(), monitor, nullptr );
        width = mode->width;
        height = mode->height;
    }
    else
    {
        const auto& rect = viewport().getViewportRect();
        // Set default windows width
        if ( width <= 0 && viewport_list.size() == 1 && MR::width( rect ) > 0 )
            width = ( int ) MR::width( rect );
        else if ( width <= 0 )
            width = 1280;
        // Set default windows height
        if ( height <= 0 && viewport_list.size() == 1 && MR::height( rect ) > 0 )
            height = ( int ) MR::height( rect );
        else if ( height <= 0 )
            height = 800;
        window = glfwCreateWindow( width, height, name.c_str(), nullptr, nullptr );
    }
    return bool( window );
}

bool Viewer::needRedraw_() const
{
    if ( dirtyScene_ )
        return true;

    for ( const auto& viewport : viewport_list )
        if ( viewport.getRedrawFlag() )
            return true;

    return getRedrawFlagRecursive( SceneRoot::get(), presentViewportsMask_ );
}

void Viewer::resetRedraw_()
{
    dirtyScene_ = false;

    for ( auto& viewport : viewport_list )
        viewport.resetRedrawFlag();

    resetRedrawFlagRecursive( SceneRoot::get() );
}

MR::Viewer::VisualObjectRenderType Viewer::getObjRenderType_( const VisualObject* obj, ViewportId viewportId ) const
{
    if ( !obj )
        return VisualObjectRenderType::Opaque;

    if ( !obj->getVisualizeProperty( VisualizeMaskType::DepthTest, viewportId ) )
        return VisualObjectRenderType::NoDepthTest;
#ifndef __EMSCRIPTEN__
    if ( auto voxObj = obj->asType<ObjectVoxels>() )
    {
        if ( voxObj->isVolumeRenderingEnabled() )
            return  VisualObjectRenderType::VolumeRendering;
    }
#endif
    if ( obj->getGlobalAlpha( viewportId ) < 255 ||
        obj->getFrontColor( obj->isSelected(), viewportId ).a < 255 ||
        obj->getBackColor( viewportId ).a < 255 )
        return VisualObjectRenderType::Transparent;

    return VisualObjectRenderType::Opaque;
}

void Viewer::recursiveDraw_( const Viewport& vp, const Object& obj, const AffineXf3f& parentXf, VisualObjectRenderType renderType, int* numDraws ) const
{
    if ( !obj.isVisible( vp.id ) )
        return;
    auto xfCopy = parentXf * obj.xf( vp.id );
    auto visObj = obj.asType<VisualObject>();
    if ( visObj && ( renderType == getObjRenderType_( visObj, vp.id ) ) )
    {
        bool alphaNeed = renderType == VisualObjectRenderType::Transparent && alphaSortEnabled_;
        vp.draw( *visObj, xfCopy, DepthFuncion::Default, alphaNeed );
        if ( numDraws )
            ++( *numDraws );
    }
    for ( const auto& child : obj.children() )
        recursiveDraw_( vp, *child, xfCopy, renderType, numDraws );
}

void Viewer::draw( bool force )
{
#ifdef __EMSCRIPTEN__
#ifdef MR_EMSCRIPTEN_ASYNCIFY
    if ( draw_( true ) )
    {
        emscripten_request_animation_frame( emsEmptyCallback, nullptr ); // call with swap
        emscripten_sleep( minEmsSleep );
    }
#else
    while ( !draw_( true ) );
#endif
#else
    draw_( force );
#endif
}

bool Viewer::draw_( bool force )
{
    bool needSceneRedraw = needRedraw_();
    if ( !force && !needSceneRedraw )
        return false;

    if ( !isInDraw_ )
        isInDraw_ = true;
    else
    {
        spdlog::error( "Recursive draw call is not allowed" );
        assert( false );
        // if this happens try to use CommandLoop instead of in draw call
        return false;
    }

    frameCounter_.startDraw();

    glPrimitivesCounter_.reset();

    setupScene();

    drawFull( needSceneRedraw );

    if ( forceRedrawFramesWithoutSwap_ > 0 )
        forceRedrawFramesWithoutSwap_--;
    auto swapped = forceRedrawFramesWithoutSwap_ == 0;

    if ( forceRedrawFrames_ > 0 )
    {
        // everything was rendered, reduce the counter
        --forceRedrawFrames_;
    }
    if ( window && swapped )
        glfwSwapBuffers( window );
    frameCounter_.endDraw( swapped );
    isInDraw_ = false;
    return ( window && swapped );
}

void Viewer::drawFull( bool dirtyScene )
{
    if ( menuPlugin_ )
        menuPlugin_->startFrame();

    if ( sceneTexture_ )
        sceneTexture_->bind( true );

    // need to clean it in texture too
    for ( auto& viewport : viewport_list )
        viewport.clearFramebuffers();

    preDrawSignal();
    // check dirty scene and need swap
    // important to check after preDrawSignal
    bool renderScene = forceRedrawFramesWithoutSwap_ <= 1;
    if ( sceneTexture_ )
        renderScene = renderScene && dirtyScene;
    if ( renderScene )
        drawScene();
    postDrawSignal();
    if ( sceneTexture_ )
    {
        sceneTexture_->unbind();
        if ( renderScene )
            sceneTexture_->copyTexture(); // copy scene texture only if scene was rendered

        sceneTexture_->draw(); // always draw scene texture
    }
    if ( menuPlugin_ )
        menuPlugin_->finishFrame();
}

void Viewer::drawScene()
{
    if ( alphaSortEnabled_ )
        alphaSorter_->clearTransparencyTextures();

    int numTransparent = 0;
    for ( auto& viewport : viewport_list )
        viewport.preDraw();

    preDrawPostViewportSignal();

    for ( const auto& viewport : viewport_list )
    {
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), VisualObjectRenderType::Opaque );
#ifndef __EMSCRIPTEN__
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), VisualObjectRenderType::VolumeRendering );
#endif
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), VisualObjectRenderType::Transparent, &numTransparent );
    }

    drawSignal();

    if ( numTransparent > 0 && alphaSortEnabled_ )
    {
        alphaSorter_->drawTransparencyTextureToScreen();
        alphaSorter_->clearTransparencyTextures();
    }
    // draw after alpha texture
    for ( const auto& viewport : viewport_list )
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), VisualObjectRenderType::NoDepthTest );

    postDrawPreViewportSignal();

    for ( const auto& viewport : viewport_list )
        viewport.postDraw();

    resetRedraw_();
}

void Viewer::setupScene()
{
    bindSceneTexture( false );
    for ( auto& viewport : viewport_list )
    {
        viewport.setupView();
        viewport.clearFramebuffers();
    }
}

void Viewer::resize( int w, int h )
{
    if ( !window )
        return;
    Vector2i fb;
    Vector2i win;
    glfwGetWindowSize( window, &win.x, &win.y );
    glfwGetFramebufferSize( window, &fb.x, &fb.y );

    Vector2f ratio( float( win.x ) / float( fb.x ), float( win.y ) / float( fb.y ) );
    glfwSetWindowSize( window, int( w * ratio.x ), int( h * ratio.y ) );
}

void Viewer::postResize( int w, int h )
{
    if ( w == 0 || h == 0 )
        return;
    if ( framebufferSize.x == w && framebufferSize.y == h )
        return;
    if ( viewport_list.size() == 1 )
    {
        ViewportRectangle rect( { 0.f, 0.f }, { float( w ), float( h ) } );
        viewport().setViewportRect( rect );
    }
    else
    {
        // It is up to the user to define the behavior of the post_resize() function
        // when there are multiple viewports (through the `callback_post_resize` callback)
        if ( w != 0 && h != 0 )
            for ( auto& viewport : viewport_list )
            {
                auto rect = viewport.getViewportRect();
                auto oldWidth = width( rect );
                auto oldHeight = height( rect );
                rect.min.x = float( rect.min.x / framebufferSize.x ) * w;
                rect.min.y = float( rect.min.y / framebufferSize.y ) * h;
                rect.max.x = rect.min.x + float( oldWidth / framebufferSize.x ) * w;
                rect.max.y = rect.min.y + float( oldHeight / framebufferSize.y ) * h;
                viewport.setViewportRect( rect );
            }
    }
    postResizeSignal( w, h );
    if ( w != 0 )
        framebufferSize.x = w;
    if ( h != 0 )
        framebufferSize.y = h;
    if ( !windowMaximized ) // resize is called after maximized
        windowSaveSize = framebufferSize;

    if ( alphaSorter_ )
        alphaSorter_->updateTransparencyTexturesSize( framebufferSize.x, framebufferSize.y );
    if ( sceneTexture_ )
        sceneTexture_->reset( framebufferSize, -1 );

#if !defined(__EMSCRIPTEN__) || defined(MR_EMSCRIPTEN_ASYNCIFY)
    if ( isLaunched_ && !isInDraw_ )
    {
        incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, true );
        while ( !draw_( true ) );
    }
#endif
}

void Viewer::postSetPosition( int xPos, int yPos )
{
    if ( !windowMaximized && !glfwGetWindowMonitor( window ) )
        windowSavePos = { xPos, yPos };
#ifdef __APPLE__
    int winWidth, winHeight;
    glfwGetWindowSize( window, &winWidth, &winHeight );
    pixelRatio = float( framebufferSize.x ) / float( winWidth );
#endif
}

void Viewer::postSetMaximized( bool maximized )
{
    windowMaximized = maximized;
    if ( windowMaximized )
        windowSavePos = windowOldPos; // maximized is called after set pos
}

void Viewer::postSetIconified( bool iconified )
{
    if ( iconified )
        windowSavePos = windowOldPos; // iconify is called after set pos
}

void Viewer::postFocus( bool focused )
{
#ifndef __EMSCRIPTEN__
    // it is needed ImGui to correctly capture events after refocusing
    if ( focused && focusRedrawReady_ && !isInDraw_ )
    {
        forceRedrawFramesWithoutSwap_ = 0;
        draw( true );
    }
#endif
    postFocusSignal( bool( focused ) );
}

void Viewer::postRescale( float x, float y )
{
    postRescaleSignal( x, y );
}

void Viewer::set_root( Object& newRoot )
{
    std::swap( SceneRoot::get(), newRoot );
}

void Viewer::initGlobalBasisAxesObject_()
{
    constexpr Vector3f PlusAxis[3] = {
        Vector3f( 1.0f, 0.0f, 0.0f ),
        Vector3f( 0.0f, 1.0f, 0.0f ),
        Vector3f( 0.0f, 0.0f, 1.0f )};

    Mesh mesh;
    globalBasisAxes = std::make_unique<ObjectMesh>();
    globalBasisAxes->setName( "World Global Basis" );
    std::vector<Color> vertsColors;
    for ( int i = 0; i < 3; ++i )
    {
        auto basis = makeCylinder( 0.01f );
        AffineXf3f rotTramsform;
        if ( i != 2 )
        {
            rotTramsform = AffineXf3f::linear(
                Matrix3f::rotation( i == 0 ? PlusAxis[1] : -1.0f * PlusAxis[0], PI_F * 0.5f )
            );
        }
        basis.transform( rotTramsform );
        mesh.addPart( basis );
        std::vector<Color> colors( basis.points.size(), Color( PlusAxis[i] ) );
        vertsColors.insert( vertsColors.end(), colors.begin(), colors.end() );
    }
    addLabel( *globalBasisAxes, "X", 1.1f * Vector3f::plusX() );
    addLabel( *globalBasisAxes, "Y", 1.1f * Vector3f::plusY() );
    addLabel( *globalBasisAxes, "Z", 1.1f * Vector3f::plusZ() );
    globalBasisAxes->setVisualizeProperty( defaultLabelsGlobalBasisAxes, VisualizeMaskType::Labels, ViewportMask::all() );
    globalBasisAxes->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    globalBasisAxes->setAncillary( true );
    globalBasisAxes->setVisible( false );
    globalBasisAxes->setVertsColorMap( std::move( vertsColors ) );
    globalBasisAxes->setColoringType( ColoringType::VertsColorMap );
}

void Viewer::initBasisAxesObject_()
{
    // store basis axes in the corner
    const float size = 0.8f;
    std::shared_ptr<Mesh> basisAxesMesh = std::make_shared<Mesh>( makeBasisAxes( size ) );
    basisAxes = std::make_unique<ObjectMesh>();
    basisAxes->setMesh( basisAxesMesh );
    basisAxes->setName("Basis axes mesh");
    basisAxes->setFlatShading( true );

    auto numF = basisAxesMesh->topology.edgePerFace().size();
    // setting color to faces
    const Color colorX = Color::red();
    const Color colorY = Color::green();
    const Color colorZ = Color::blue();
    FaceColors colorMap( numF );
    const auto arrowSize = numF / 3;
    for (int i = 0; i < arrowSize; i++)
    {
        colorMap[FaceId( i )] = colorX;
        colorMap[FaceId( i + arrowSize )] = colorY;
        colorMap[FaceId( i + arrowSize * 2 )] = colorZ;
    }
    const float labelPos = size + 0.2f;

    addLabel( *basisAxes, "X", labelPos * Vector3f::plusX() );
    addLabel( *basisAxes, "Y", labelPos * Vector3f::plusY() );
    addLabel( *basisAxes, "Z", labelPos * Vector3f::plusZ() );

    basisAxes->setVisualizeProperty( defaultLabelsBasisAxes, VisualizeMaskType::Labels, ViewportMask::all() );
    basisAxes->setFacesColorMap( colorMap );
    basisAxes->setColoringType( ColoringType::FacesColorMap );
}

void Viewer::initClippingPlaneObject_()
{
    std::shared_ptr<Mesh> plane = std::make_shared<Mesh>( makePlane() );
    clippingPlaneObject = std::make_unique<ObjectMesh>();
    clippingPlaneObject->setMesh( plane );
    clippingPlaneObject->setName( "Clipping plane obj" );
    clippingPlaneObject->setVisible( false );
    clippingPlaneObject->setFrontColor( Color( Vector4f::diagonal( 0.2f ) ), false );
    clippingPlaneObject->setBackColor( Color( Vector4f::diagonal( 0.2f ) ) );
}

void Viewer::initRotationCenterObject_()
{
    constexpr Color color = Color( 0, 127, 0, 255 );
    auto mesh = makeUVSphere();
    rotationSphere = std::make_unique<ObjectMesh>();
    rotationSphere->setFrontColor( color, false );
    rotationSphere->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    rotationSphere->setAncillary( true );
}

void Viewer::initSpaceMouseHandler_()
{
    

    #if defined(__EMSCRIPTEN__)
        spaceMouseHandler_ = std::make_unique<SpaceMouseHandler>();
    #else
        spaceMouseHandler_ = std::make_unique<SpaceMouseHandlerHidapi>();
    #endif

    spaceMouseHandler_->initialize();
}

bool Viewer::windowShouldClose()
{
    if ( !( window && glfwWindowShouldClose( window ) ) && !stopEventLoop_ )
        return false;

    if ( !interruptWindowClose() )
        return true;

    if ( window )
        glfwSetWindowShouldClose( window, false );
    stopEventLoop_ = false;
    return false;
}

void Viewer::makeTitleFromSceneRootPath()
{
    auto sceneFileName = utf8string( SceneRoot::getScenePath().filename() );
    if ( globalHistoryStore_ && globalHistoryStore_->isSceneModified() )
        sceneFileName += "*";

    if ( sceneFileName.empty() )
        glfwSetWindowTitle( window, defaultWindowTitle.c_str() );
    else
        glfwSetWindowTitle( window, (defaultWindowTitle + " " + sceneFileName).c_str() );
}

ViewportId Viewer::getFirstAvailableViewportId_() const
{
    ViewportId res{1};
    while ( res.valid() )
    {
        if ( ( presentViewportsMask_ & res ).empty() )
            return res;
        res = res.next();
    }
    return res;
}

void Viewer::clearScene()
{
    SceneRoot::get().removeAllChildren();
}

MR::Viewport& Viewer::viewport( ViewportId viewport_id )
{
    assert( !viewport_list.empty() && "viewport_list should never be empty" );
    int viewport_index;
    if ( !viewport_id.valid() )
        viewport_index = (int) selected_viewport_index;
    else
        viewport_index = (int) this->viewport_index( viewport_id );
    assert( ( viewport_index >= 0 && viewport_index < viewport_list.size() ) && "selected_viewport_index should be in bounds" );
    return viewport_list[viewport_index];
}

const MR::Viewport& Viewer::viewport( ViewportId viewport_id ) const
{
    assert( !viewport_list.empty() && "viewport_list should never be empty" );
    int viewport_index;
    if ( !viewport_id.valid() )
        viewport_index = (int) selected_viewport_index;
    else
        viewport_index = (int) this->viewport_index( viewport_id );
    assert( ( viewport_index >= 0 && viewport_index < viewport_list.size() ) && "selected_viewport_index should be in bounds" );
    return viewport_list[viewport_index];
}

ViewportId Viewer::append_viewport( Viewport::ViewportRectangle viewportRect, bool append_empty /*= false */ )
{
    auto nextId = getFirstAvailableViewportId_();
    if ( !nextId )
    {
        spdlog::error( "No ViewportId available " );
        return nextId;
    }

    viewport_list.push_back( viewport().clone() ); // copies the previous active viewport and only changes the viewport
    viewport_list.back().id = nextId;
    viewport_list.back().init();
    viewport_list.back().setViewportRect( viewportRect );
    if ( append_empty )
    {
        for ( const auto& child : SceneRoot::get().children() )
            child->setVisible( false, nextId );
    }
    selected_viewport_index = viewport_list.size() - 1;
    presentViewportsMask_ |= nextId;
    return viewport_list.back().id;
}

Box2f Viewer::getViewportsBounds() const
{
    Box2f box;
    for ( const auto& vp : viewport_list )
    {
        const auto& rect = vp.getViewportRect();
        box.include( Box2f::fromMinAndSize( { rect.min.x,rect.min.y }, { width( rect ),height( rect ) } ) );
    }
    return box;
}

bool Viewer::erase_viewport( const size_t index )
{
    assert( index < viewport_list.size() && "index should be in bounds" );
    if ( viewport_list.size() == 1 )
    {
        // Cannot remove last viewport
        return false;
    }
    viewport_list[index].shut(); // does nothing
    presentViewportsMask_ &= ~ViewportMask( viewport_list[index].id );
    viewport_list.erase( viewport_list.begin() + index );
    if ( selected_viewport_index >= index && selected_viewport_index > 0 )
    {
        selected_viewport_index--;
    }
    return true;
}

bool Viewer::erase_viewport( ViewportId viewport_id )
{
    auto index = viewport_index( viewport_id );
    if ( index < 0 )
        return false;
    return erase_viewport( index );
}

int Viewer::viewport_index( const ViewportId id ) const
{
    for ( int i = 0; i < viewport_list.size(); ++i )
    {
        if ( viewport_list[i].id == id )
            return i;
    }
    return -1;
}

ViewportId Viewer::getHoveredViewportId() const
{
    const auto& currentPos = mouseController.getMousePos();
    for ( int i = 0; i < viewport_list.size(); i++ )
    {
        if ( !viewport_list[i].getParameters().selectable )
            continue;

        const auto& rect = viewport_list[i].getViewportRect();

        if ( ( currentPos.x > rect.min.x ) &&
             ( currentPos.x < rect.min.x + width( rect ) ) &&
             ( ( framebufferSize.y - currentPos.y ) > rect.min.y ) &&
             ( ( framebufferSize.y - currentPos.y ) < rect.min.y + height( rect ) ) )
        {
            return viewport_list[i].id;
        }
    }

    return viewport_list[selected_viewport_index].id;
}

void Viewer::select_hovered_viewport()
{
    selected_viewport_index = viewport_index( getHoveredViewportId() );
}

void Viewer::fitDataViewport( MR::ViewportMask vpList, float fill, bool snapView )
{
    for( auto& viewport : viewport_list )
    {
        if( viewport.id.value() & vpList.value() )
        {
            viewport.fitData( fill, snapView );
        }
    }
}

void Viewer::fitBoxViewport( const Box3f& box, MR::ViewportMask vpList /*= MR::ViewportMask::all()*/, float fill /*= 0.6f*/, bool snapView /*= true */ )
{
    for ( auto& viewport : viewport_list )
    {
        if ( viewport.id.value() & vpList.value() )
        {
            viewport.fitBox( box, fill, snapView );
        }
    }
}

void Viewer::preciseFitDataViewport( MR::ViewportMask vpList, const Viewport::FitDataParams& params )
{
    for( auto& viewport : viewport_list )
    {
        if( viewport.id.value() & vpList.value() )
        {
            viewport.preciseFitDataToScreenBorder( params );
        }
    }
}

void Viewer::incrementForceRedrawFrames( int i /*= 1 */, bool swapOnLastOnly /*= false */)
{
    if ( isInDraw_ )
        ++i;
    forceRedrawFrames_ = std::max( i, forceRedrawFrames_ );
    if ( swapOnLastOnly )
        forceRedrawFramesWithoutSwap_ = std::max( i, forceRedrawFramesWithoutSwap_ );
}

bool Viewer::isCurrentFrameSwapping() const
{
    return forceRedrawFramesWithoutSwap_ == 0;
}

size_t Viewer::getEventsCount( EventType type ) const
{
    return eventsCounter_.counter[size_t( type )];
}

size_t Viewer::getLastFrameGLPrimitivesCount( GLPrimitivesType type ) const
{
    return glPrimitivesCounter_.counter[size_t( type )];
}

void Viewer::incrementThisFrameGLPrimitivesCount( GLPrimitivesType type, size_t num )
{
    glPrimitivesCounter_.counter[size_t( type )] += num;
}

void Viewer::resetAllCounters()
{
    eventsCounter_.reset();
    frameCounter_.reset();
}

Image Viewer::captureSceneScreenShot( const Vector2i& resolution )
{
    if ( !glInitialized_ )
        return {};

    Vector2i newRes;
    newRes.x = resolution.x <= 0 ? framebufferSize.x : resolution.x;
    newRes.y = resolution.y <= 0 ? framebufferSize.y : resolution.y;

    // store old sizes
    auto vpBounbds = getViewportsBounds();
    std::vector<ViewportRectangle> rects;
    for ( auto& viewport : viewport_list )
    {
        auto rect = viewport.getViewportRect();
        rects.push_back( rect );
        rect.min.x = float( rect.min.x - vpBounbds.min.x ) / width( vpBounbds ) * newRes.x;
        rect.min.y = float( rect.min.y - vpBounbds.min.y ) / height( vpBounbds ) * newRes.y;
        rect.max.x = float( rect.max.x - vpBounbds.min.x ) / width( vpBounbds ) * newRes.x;
        rect.max.y = float( rect.max.y - vpBounbds.min.y ) / height( vpBounbds ) * newRes.y;
        viewport.setViewportRect( rect );
    }
    if ( newRes != framebufferSize && alphaSorter_ )
        alphaSorter_->updateTransparencyTexturesSize( newRes.x, newRes.y );


    std::vector<Color> pixels( newRes.x * newRes.x );

    FramebufferData fd;
    fd.gen( newRes, true );
    fd.bind();

    setupScene();
    drawScene();

    fd.copyTextureBindDef();
    fd.bindTexture();

    GL_EXEC( glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ( void* )( pixels.data() ) ) );
    
    fd.del();

    bindSceneTexture( true );

    // restore sizes
    int i = 0;
    for ( auto& viewport : viewport_list )
        viewport.setViewportRect( rects[i++] );
    if ( newRes != framebufferSize && alphaSorter_ )
        alphaSorter_->updateTransparencyTexturesSize( framebufferSize.x, framebufferSize.y );

    return Image{ pixels, newRes };
}

void Viewer::captureUIScreenShot( std::function<void( const Image& )> callback,
                                  const Vector2i& pos /*= Vector2i()*/, const Vector2i& sizeP /*= Vector2i() */ )
{
    CommandLoop::appendCommand( [callback, pos, sizeP, this] ()
    {
        Vector2i size = sizeP;
        if ( !size.x )
            size.x = framebufferSize.x - pos.x;
        else
            size.x = std::min( framebufferSize.x - pos.x, size.x );

        if ( !size.y )
            size.y = framebufferSize.y - pos.y;
        else
            size.y = std::min( framebufferSize.y - pos.y, size.y );

        Image image;
        image.resolution = size;
        image.pixels.resize( size.x * size.x );

        if ( glInitialized_ )
        {
            GL_EXEC( glReadPixels( pos.x, pos.y, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, ( void* ) ( image.pixels.data() ) ) );

            callback( image );
        }
    } );
}

bool Viewer::isAlphaSortAvailable() const
{
    return bool( alphaSorter_ );
}

bool Viewer::enableAlphaSort( bool on )
{
    if ( on == alphaSortEnabled_ )
        return false;
    if ( !on )
    {
        alphaSortEnabled_ = false;
        return true;
    }

    if ( !isAlphaSortAvailable() )
        return false;

    alphaSortEnabled_ = true;
    return true;
}

void Viewer::bindSceneTexture( bool bind )
{
    if ( !sceneTexture_ )
        return;
    if ( bind )
        sceneTexture_->bind( false );
    else
        sceneTexture_->unbind();
}

void Viewer::setViewportSettingsManager( std::unique_ptr<IViewerSettingsManager> mng )
{
    settingsMng_ = std::move( mng );
}

Viewer::PointInAllSpaces Viewer::getMousePointInfo() const
{
    const auto& currentPos = mouseController.getMousePos();
    return getPixelPointInfo( Vector3f( float( currentPos.x ), float( currentPos.y ), 0.f ) );
}

Viewer::PointInAllSpaces Viewer::getPixelPointInfo( const Vector3f& screenPoint ) const
{
    PointInAllSpaces res;
    res.screenSpace = screenPoint;
    for( const auto& viewport : viewport_list )
    {
        res.viewportSpace = screenToViewport( screenPoint, viewport.id );
        const auto& rect = viewport.getViewportRect();
        if ( res.viewportSpace.x > 0 && res.viewportSpace.x < width( rect ) &&
             res.viewportSpace.y > 0 && res.viewportSpace.y < height( rect ) )
        {
            res.viewportId = viewport.id;
            res.clipSpace = viewport.viewportSpaceToClipSpace(
                Vector3f( static_cast<float>(res.viewportSpace.x), static_cast<float>(res.viewportSpace.y), 0.f )
            );
            // looking for all visible objects
            auto [obj, pick] = viewport.pick_render_object( Vector2f( res.viewportSpace.x, res.viewportSpace.y ) );
            if(obj)
            {
                res.obj = obj;
                res.pof = pick;
                res.worldSpace = res.obj->worldXf()( pick.point );
                res.cameraSpace = viewport.worldToCameraSpace( res.worldSpace );
                res.clipSpace.z = viewport.projectToClipSpace( res.worldSpace ).z;
            }
            return res;
        }
    }
    return PointInAllSpaces();
}

Vector3f Viewer::screenToViewport( const Vector3f& screenPoint, ViewportId id ) const
{
    if( (presentViewportsMask_ & id).empty() )
        return { 0.f, 0.f, 0.f };

    const auto& rect = viewport( id ).getViewportRect();
    return { screenPoint.x - rect.min.x, screenPoint.y + rect.min.y + height( rect ) - framebufferSize.y, screenPoint.z };
}

Vector3f Viewer::viewportToScreen( const Vector3f& viewportPoint, ViewportId id ) const
{
    if( (presentViewportsMask_ & id).empty() )
        return { 0.f, 0.f, 0.f };

    const auto& rect = viewport( id ).getViewportRect();
    return { viewportPoint.x + rect.min.x, viewportPoint.y - rect.min.y - height( rect ) + framebufferSize.y, viewportPoint.z };
}

std::vector<std::reference_wrapper<Viewport>> Viewer::getViewports( ViewportMask mask )
{
    std::vector<std::reference_wrapper<Viewport>> res;
    for( auto& viewport : viewport_list )
    {
        if( viewport.id.value() & mask.value() )
        {
            res.push_back( viewport );
        }
    }
    return res;
}

void Viewer::enableGlobalHistory( bool on )
{
    if ( on == bool( globalHistoryStore_ ) )
        return;
    if ( on )
        globalHistoryStore_ = std::make_shared<HistoryStore>();
    else
        globalHistoryStore_.reset();
}

void Viewer::appendHistoryAction( const std::shared_ptr<HistoryAction>& action )
{
    if ( globalHistoryStore_ )
    {
        globalHistoryStore_->appendAction( action );
        makeTitleFromSceneRootPath();
    }
}

bool Viewer::globalHistoryUndo()
{
    if ( globalHistoryStore_ && globalHistoryStore_->undo() )
    {
        makeTitleFromSceneRootPath();
        return true;
    }
    return false;
}

bool Viewer::globalHistoryRedo()
{
    if ( globalHistoryStore_ && globalHistoryStore_->redo() )
    {
        makeTitleFromSceneRootPath();
        return true;
    }
    return false;
}

void Viewer::onSceneSaved( const std::filesystem::path& savePath, bool storeInRecent )
{
    if ( !savePath.empty() && storeInRecent )
        recentFilesStore.storeFile( savePath );

    if ( !SceneFileFilters.empty() && savePath.extension() == SceneFileFilters.front().extensions.substr( 1 ) )
        SceneRoot::setScenePath(savePath);
    else
        SceneRoot::setScenePath("");

    if ( globalHistoryStore_ )
        globalHistoryStore_->setSavedState();

    makeTitleFromSceneRootPath();
}

const std::shared_ptr<ImGuiMenu>& Viewer::getMenuPlugin() const
{
    return menuPlugin_;
}

void Viewer::setMenuPlugin( std::shared_ptr<ImGuiMenu> menu )
{
    assert( !menuPlugin_ );
    assert( menu );
    menuPlugin_ = menu;
}

size_t Viewer::getStaticGLBufferSize() const
{
    return GLStaticHolder::getStaticGLBuffer().heapBytes();
}

void Viewer::FrameCounter::startDraw()
{
    startDrawTime_ = std::chrono::high_resolution_clock::now();
}

void Viewer::FrameCounter::endDraw( bool swapped )
{
    ++totalFrameCounter;
    if ( swapped )
    {
        ++swappedFrameCounter;
        const auto nowTP = std::chrono::high_resolution_clock::now();
        const auto nowSec = std::chrono::time_point_cast<std::chrono::seconds>( nowTP ).time_since_epoch().count();
        drawTimeMilliSec =  ( nowTP - startDrawTime_ ) * 1000;
        if ( nowSec > startFPSTime_ )
        {
            startFPSTime_ = nowSec;
            fps = swappedFrameCounter - startFrameNum;
            startFrameNum = swappedFrameCounter;
        }
    }
}

void Viewer::FrameCounter::reset()
{
    totalFrameCounter = 0;
    swappedFrameCounter = 0;
    startFPSTime_ = 0;
    fps = 0;
    startFrameNum = 0;
}

void Viewer::EventsCounter::reset()
{
    for ( size_t i = 0; i < size_t( EventType::Count ); ++i )
        counter[i] = 0;
}

void Viewer::GLPrimitivesCounter::reset()
{
    for ( size_t i = 0; i < size_t( GLPrimitivesType::Count ); ++i )
        counter[i] = 0;
}

// simple test to make sure this dll was linked and loaded to test project
TEST( MRViewer, LoadTest )
{
    bool load = true;
    ASSERT_EQ( load, true );
}


}
