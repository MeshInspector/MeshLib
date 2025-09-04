#include "MRViewer.h"
#include "MRViewerEventQueue.h"
#include "MRSceneTextureGL.h"
#include "MRAlphaSortGL.h"
#include "MRGLMacro.h"
#include "MRSetupViewer.h"
#include "MRGLStaticHolder.h"
#include "MRViewerPlugin.h"
#include "MRCommandLoop.h"
#include "MRSplashWindow.h"
#include "MRViewerSettingsManager.h"
#include "MRGladGlfw.h"
#include "MRRibbonMenu.h"
#include "MRGetSystemInfoJson.h"
#include "MRSpaceMouseHandler.h"
#include "MRDragDropHandler.h"
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRSpaceMouseHandler3dxMacDriver.h"
#include "MRRenderGLHelpers.h"
#include "MRTouchpadController.h"
#include "MRSpaceMouseController.h"
#include "MRTouchesController.h"
#include "MRMouseController.h"
#include "MRRecentFilesStore.h"
#include "MRPointInAllSpaces.h"
#include "MRViewport.h"
#include "MRFrameCounter.h"
#include "MRColorTheme.h"
#include "MRHistoryStore.h"
#include "MRShowModal.h"
#include "MRFileDialog.h"
#include <MRMesh/MRFinally.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRArrow.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MRToFromEigen.h>
#include <MRMesh/MRTimer.h>
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsLoad.h"
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
#include "MRMesh/MRGTest.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRSceneColors.h"
#include "MRPch/MRWasm.h"
#include "MRMesh/MRGcodeLoad.h"
#include "MRSceneCache.h"
#include "MRViewerTitle.h"
#include "MRViewportCornerController.h"
#include "MRViewportGlobalBasis.h"
#include "MRWebRequest.h"
#include "MRMesh/MRCube.h"

#ifndef __EMSCRIPTEN__
#include <boost/exception/diagnostic_information.hpp>
#endif
#include "MRSaveObjects.h"
#include "MRProgressBar.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRAppendHistory.h"
#include "MRSwapRootAction.h"
#include "MRMesh/MRSceneLoad.h"
#include "MRMesh/MRConfig.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/html5.h>
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
    auto& settingsManager = viewer.getViewerSettingsManager();
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
    viewer->emplaceEvent( "Mouse press", [mb, action, modifier, viewer] ()
    {
        if ( action == GLFW_PRESS )
            viewer->mouseDown( mb, modifier );
        else
            viewer->mouseUp( mb, modifier );
    } );
}

static void glfw_error_callback( int /*error*/, const char* description )
{
    spdlog::error( "glfw_error_callback: {}", description );
}

static void glfw_char_mods_callback( GLFWwindow* /*window*/, unsigned int codepoint )
{
    auto viewer = &MR::getViewerInstance();
    viewer->emplaceEvent( "Char", [codepoint, viewer] ()
    {
        viewer->keyPressed( codepoint, 0 );
    } );
}

static void glfw_key_callback( GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int modifier )
{
    auto viewer = &MR::getViewerInstance();
    viewer->emplaceEvent( "Key press", [action, key, modifier, viewer] ()
    {
        if ( action == GLFW_PRESS )
            viewer->keyDown( key, modifier );
        else if ( action == GLFW_RELEASE )
            viewer->keyUp( key, modifier );
        else if ( action == GLFW_REPEAT )
            viewer->keyRepeat( key, modifier );
    } );
}

static bool gWindowSizeInitialized = false;

static void glfw_framebuffer_size( GLFWwindow* /*window*/, int width, int height )
{
    auto viewer = &MR::getViewerInstance();
#if defined( __linux__ ) && !defined( __EMSCRIPTEN__ )
    if ( gWindowSizeInitialized )
    {
        // on Linux some (or all?) window managers send resize events in batch, processing all of them hits performance
        viewer->emplaceEvent( "Window resize", [width, height, viewer]
        {
            viewer->postResize( width, height );
        }, true );
    }
    else
#endif
    {
        viewer->postResize( width, height );
        viewer->postEmptyEvent();
    }
}

static void glfw_window_pos( GLFWwindow* /*window*/, int xPos, int yPos )
{
    // need for remember window pos and size before maximize
    // located here because glfw_window_pos calls before glfw_window_maximize and glfw_window_iconify (experience)
    auto viewer = &MR::getViewerInstance();
    viewer->emplaceEvent( "Windows pos", [xPos, yPos, viewer] ()
    {
        viewer->windowOldPos = viewer->windowSavePos;
        viewer->postSetPosition( xPos, yPos );
    } );
}

static void glfw_cursor_enter_callback( GLFWwindow* /*window*/, int entered )
{
    auto viewer = &MR::getViewerInstance();
    viewer->emplaceEvent( "Cursor enter", [entered, viewer] ()
    {
        viewer->cursorEntranceSignal( bool( entered ) );
    } );
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

static void glfw_window_close( GLFWwindow* /*window*/ )
{
    // needed not to sleep until next event on close
    MR::getViewerInstance().postClose();
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
    viewer->emplaceEvent( "Mouse move", eventCall, true );
}

static void glfw_mouse_scroll( GLFWwindow* /*window*/, double /*x*/, double y )
{
    static double prevY = 0.0;
    auto viewer = &MR::getViewerInstance();
    if ( prevY * y < 0.0 )
        viewer->popEventByName( "Mouse scroll" );
    auto eventCall = [y, viewer, prevPtr = &prevY] ()
    {
        *prevPtr = y;
        viewer->mouseScroll( float( y ) );
        viewer->draw();
    };
    viewer->emplaceEvent( "Mouse scroll", eventCall );
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
    viewer->emplaceEvent( "Drop", [paths, viewer] ()
    {
        viewer->dragDrop( paths );
    } );
    viewer->postEmptyEvent();
}

namespace MR
{

void Viewer::emplaceEvent( std::string name, ViewerEventCallback cb, bool skipable )
{
    if ( eventQueue_ )
        eventQueue_->emplace( std::move( name ), std::move( cb ), skipable );
}

void Viewer::popEventByName( const std::string& name )
{
    if ( eventQueue_ )
        eventQueue_->popByName( name );
}

void addLabel( ObjectMesh& obj, const std::string& str, const Vector3f& pos, bool depthTest )
{
    auto label = std::make_shared<ObjectLabel>();
    label->setFrontColor( Color::white(), false );
    label->setLabel( { str, pos } );
    label->setPivotPoint( Vector2f( 0.5f, 0.5f ) );
    label->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
    float fontSize = 20.0f;
    if ( auto menu = getViewerInstance().getMenuPlugin() )
        fontSize *= menu->menu_scaling();
    label->setFontHeight( fontSize );
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

    CommandLoop::setMainThreadId( std::this_thread::get_id() );

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

    int res = 0;
#if defined(__EMSCRIPTEN__) || !defined(NDEBUG)
    res = viewer.launch( params );
#else
    try
    {
        res = viewer.launch( params );
    }
    catch ( ... )
    {
        spdlog::critical( boost::current_exception_diagnostic_information() );
        spdlog::info( "Exception stacktrace:\n{}", getCurrentStacktrace() );
        printCurrentTimerBranch();
        res = 1;
    }
#endif
    if ( params.unloadPluginsAtEnd )
        setup.unloadExtendedLibraries();
    return res;
}

void loadMRViewerDll()
{
}

void filterReservedCmdArgs( std::vector<std::string>& args )
{
    if ( args.empty() )
        return;
    bool nextW{ false };
    bool nextH{ false };
    bool nextFPS{ false };
    std::vector<int> indicesToRemove;
    indicesToRemove.push_back( 0 );
    for ( int i = 1; i < args.size(); ++i )
    {
        bool reserved = false;
        const auto& flag = args[i];
        if ( nextW )
        {
            nextW = false;
            reserved = true;
        }
        else if ( nextH )
        {
            nextH = false;
            reserved = true;
        }
        else if ( nextFPS )
        {
            nextFPS = false;
            reserved = true;
        }
        else if (
            flag == "-noWindow" ||
            flag == "-fullscreen" ||
            flag == "-noClose" ||
            flag == "-noEventLoop" ||
            flag == "-hidden" ||
            flag == "-tryHidden" ||
            flag == "-transparentBgOn" ||
            flag == "-transparentBgOff" ||
            flag == "-noSplash" ||
    #if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
            flag == "-showSplash" ||
    #endif
            flag == "-console" ||
            flag == "-openGL3" ||
            flag == "-noRenderInTexture" ||
            flag == "-develop" ||
            flag == "-unloadPluginsAtEnd"
            )
            reserved = true;
        else if ( flag == "-width" )
        {
            nextW = true;
            reserved = true;
        }
        else if ( flag == "-height" )
        {
            nextH = true;
            reserved = true;
        }
        else if ( flag == "-animateFPS" )
        {
            nextFPS = true;
            reserved = true;
        }

        if ( reserved )
            indicesToRemove.push_back( i );
    }
    for ( int i = int( indicesToRemove.size() ) - 1; i >= 0; --i )
        args.erase( args.begin() + indicesToRemove[i] );
}

void Viewer::parseLaunchParams( LaunchParams& params )
{
    bool nextW{ false };
    bool nextH{ false };
    bool nextFPS{ false };
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
        else if ( nextFPS )
        {
            nextFPS = false;
            auto fps = std::atoi( flag.c_str() );
            if ( fps > 0 )
            {
                params.isAnimating = true;
                params.animationMaxFps = fps;
            }
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
    #if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
        else if ( flag == "-showSplash" )
            params.splashWindow = std::make_shared<MR::DefaultSplashWindow>();
    #endif
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
        else if ( flag == "-animateFPS" )
            nextFPS = true;
        else if ( flag == "-unloadPluginsAtEnd" )
            params.unloadPluginsAtEnd = true;
    }
}

#ifdef __EMSCRIPTEN__
#ifndef MR_EMSCRIPTEN_ASYNCIFY
void Viewer::emsMainInfiniteLoop()
{
    auto& viewer = getViewerInstance();
    viewer.draw( true );
    if ( viewer.eventQueue_ )
        viewer.eventQueue_->execute();
    CommandLoop::processCommands();
}
#else
void Viewer::emsMainInfiniteLoop()
{
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
        else if ( !isAnimating && eventQueue_ && eventQueue_->empty() )
        {
            emscripten_sleep( minEmsSleep ); // more then 300 fps possible
            continue;
        }

        do
        {
            draw( true );
            if ( eventQueue_ )
                eventQueue_->execute();
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
    commandArgs.resize( params.argc );
    for ( int i = 0; i < params.argc; ++i )
    {
        commandArgs[i] = std::string( params.argv[i] );
        spdlog::info( "argv[{}]: {}", i, commandArgs[i] );
    }
    filterReservedCmdArgs( commandArgs );

    launchParams_ = params;
    isAnimating = params.isAnimating;
    animationMaxFps = params.animationMaxFps;
    if ( params.developerFeatures )
        experimentalFeatures = true;
    auto res = launchInit_( params );
    if ( res != EXIT_SUCCESS )
        return res;

    CommandLoop::setState( CommandLoop::StartPosition::AfterSplashHide );
    CommandLoop::processCommands(); // execute pre init commands before first draw
    focusRedrawReady_ = true;

    if ( params.windowMode == LaunchParams::HideInit && window )
        glfwShowWindow( window );

    CommandLoop::setState( CommandLoop::StartPosition::AfterWindowAppear );
    CommandLoop::processCommands(); // execute remaining commands in the queue, important for params.startEventLoop==false

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

    CommandLoop::removeCommands( true );

    return EXIT_SUCCESS;
}

bool Viewer::checkOpenGL_( const LaunchParams& params )
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
            if ( params.windowMode != LaunchParams::WindowMode::TryHidden )
            {
                MessageBoxA( NULL, "Cannot activate OpenGL 3.3.\n"
                    "Please verify that you have decent graphics card and its drivers are installed.",
                    "MeshInspector/MeshLib Error", MB_OK );
            }
#endif
            return false;
        }
        spdlog::warn( "Alpha sort is not available" );
    }
#endif
#endif
    return true;
}

bool Viewer::setupWindow_( const LaunchParams& params )
{
    MR_TIMER;
    assert( window );
    spdlog::info( "Setting up window" );

    glfwMakeContextCurrent( window );
    if ( !loadGL() )
    {
        spdlog::error( "Failed to load OpenGL and its extensions" );
        return false;
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

    if ( !windowTitle )
        windowTitle = std::make_shared<ViewerTitle>();

    windowTitle->setAppName( params.name );
    if ( params.showMRVersionInTitle )
        windowTitle->setVersion( GetMRVersionString() );

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
    glfwSetWindowCloseCallback( window, glfw_window_close );
#endif
    glfwSetMouseButtonCallback( window, glfw_mouse_press );
    glfwSetCharCallback( window, glfw_char_mods_callback );
    glfwSetDropCallback( window, glfw_drop_callback );

    // Handle retina displays (windows and mac)
    int width, height;
    glfwGetFramebufferSize( window, &width, &height );
    // Initialize IGL viewer
    glfw_framebuffer_size( window, width, height );

    if ( hasScaledFramebuffer_ )
        updatePixelRatio_();

    float xscale{ 1.0f }, yscale{ 1.0f };
#ifndef __EMSCRIPTEN__
    glfwGetWindowContentScale( window, &xscale, &yscale );
#endif
    glfw_window_scale( window, xscale, yscale );
    spdlog::info( "glfw configured" );

    enableAlphaSort( true );
    if ( sceneTexture_ )
    {
        sceneTexture_->reset( { width, height }, getMSAAPow( getRequiredMSAA_( true, true ) ) );
        spdlog::info( "SceneTexture created" );
    }

    if ( alphaSorter_ )
    {
        alphaSorter_->init();
        alphaSorter_->updateTransparencyTexturesSize( width, height );
        spdlog::info( "AlphaSorter created" );
    }

    mouseController_->connect();

    if ( !touchesController_ )
        touchesController_ = std::make_unique<TouchesController>();
    touchesController_->connect( this );
    spdlog::info( "TouchesController created" );

    if ( !spaceMouseController_ )
        spaceMouseController_ = std::make_unique<SpaceMouseController>();
    spaceMouseController_->connect();
    spdlog::info( "SpaceMouseController created" );

    if ( !spaceMouseHandler_ )
        initSpaceMouseHandler();

    if ( !touchpadController_ )
        touchpadController_ = std::make_unique<TouchpadController>();
    touchpadController_->connect( this );
    touchpadController_->initialize( window );
    spdlog::info( "TouchpadController created" );

    dragDropAdvancedHandler_ = getDragDropHandler( window );
    return true;
}

int Viewer::launchInit_( const LaunchParams& params )
{
    MR_TIMER;
    CommandLoop::setMainThreadId( std::this_thread::get_id() );
    spdlog::info( "Log file: {}", utf8string( Logger::instance().getLogFileName() ) );
    glfwSetErrorCallback( glfw_error_callback );

#if GLFW_VERSION_MAJOR > 3 || ( GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4 )
#if !defined( MRVIEWER_EXPERIMENTAL_WAYLAND_SUPPORT ) && defined( __linux__ )
    // force to use X11
    if ( glfwPlatformSupported( GLFW_PLATFORM_X11 ) )
        glfwInitHint( GLFW_PLATFORM, GLFW_PLATFORM_X11 );
#endif
#endif

    if ( !glfwInit() )
    {
        spdlog::error( "glfwInit failed" );
        return EXIT_FAILURE;
    }
    spdlog::info( "glfwInit succeeded" );

#if defined( __APPLE__ )
    // Setting window properties
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE );
    glfwWindowHint( GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE ); // required for GLFW 3.3
    hasScaledFramebuffer_ = true;
#else
#if GLFW_VERSION_MAJOR > 3 || ( GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4 )
    if ( glfwGetPlatform() == GLFW_PLATFORM_WAYLAND )
    {
        glfwWindowHintString( GLFW_WAYLAND_APP_ID, MR_PROJECT_NAME );
        glfwWindowHint( GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE );
        hasScaledFramebuffer_ = true;
    }
#endif
#endif
    glfwWindowHint( GLFW_SAMPLES, getRequiredMSAA_( params.render3dSceneInTexture, false ) );
#ifndef __EMSCRIPTEN__
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_FOCUS_ON_SHOW, GLFW_TRUE );
    glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, params.enableTransparentBackground );
#endif

    glfwWindowHint( GLFW_VISIBLE, int( bool( params.windowMode == LaunchParams::Show ) ) );
    bool windowMode = params.windowMode != LaunchParams::NoWindow;

    if ( windowMode )
    {
        alphaSorter_ = std::make_unique<AlphaSortGL>();
        if ( params.render3dSceneInTexture )
            sceneTexture_ = std::make_unique<SceneTextureGL>();

        // Create the window
        if ( !checkOpenGL_( params ) )
        {
            if ( params.windowMode == LaunchParams::TryHidden )
                windowMode = false;
            else
                return EXIT_FAILURE;
        }
    }

    if ( windowMode && !setupWindow_( params ) )
        return -1;

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
    *recentFilesStore_ = RecentFilesStore( params.name );

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

    spdlog::info( "Launching event loop" );

    // Rendering loop
    while ( !windowShouldClose() )
    {
        do
        {
            draw( true );
            glfwPollEvents();
            if ( eventQueue_ )
                eventQueue_->execute();
            if ( spaceMouseHandler_ )
                spaceMouseHandler_->handle();
            CommandLoop::processCommands();
        } while ( ( !( window && glfwWindowShouldClose( window ) ) && !stopEventLoop_ ) && ( forceRedrawFrames_ > 0 || needRedraw_() ) );

        if ( isAnimating )
        {
            const double minDuration = 1.0 / double( animationMaxFps );
            glfwWaitEventsTimeout( minDuration );
            if ( eventQueue_ )
                eventQueue_->execute();
        }
        else
        {
            glfwWaitEvents();
            if ( eventQueue_ )
                eventQueue_->execute();
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

    {
        spdlog::info( "Wait and process unfinished web requests." );
        /// wait for all remaining requests
        for ( int i = 0; i < 3; ++i ) // maximum 3 iterations
        {
            WebRequest::waitRemainingAsync();
            if ( CommandLoop::empty() )
                break;
            CommandLoop::processCommands();
        }
    }

    for ( auto& viewport : viewport_list )
        viewport.shut();
    shutdownPlugins_();

    // Clear plugins
    plugins.clear();
    menuPlugin_.reset();

    // Clear objects
    SceneRoot::get().removeAllChildren();
    basisAxes.reset();
    rotationSphere.reset();
    clippingPlaneObject.reset();
    globalBasis.reset();
    globalHistoryStore_.reset();
    basisViewController.reset();

    GLStaticHolder::freeAllShaders();

    alphaSorter_.reset();
    sceneTexture_.reset();

    if ( touchpadController_ )
        touchpadController_->reset();

    dragDropAdvancedHandler_.reset();

    glfwDestroyWindow( window );
    glfwTerminate();
    glInitialized_ = false;
    isLaunched_ = false;
    spaceMouseHandler_.reset();

    /// removes references on all cached objects before shared libraries with plugins are unloaded
    SceneCache::invalidateAll();

    {
        // some requests might be sent during shutdown, just wait for them but don't process
        spdlog::info( "Wait and DON'T process unfinished web requests." );
        WebRequest::waitRemainingAsync();
    }

    /// disconnect all slots before shared libraries with plugins are unloaded
    mouseDownSignal = {};
    mouseUpSignal = {};
    mouseMoveSignal = {};
    mouseScrollSignal = {};
    mouseClickSignal = {};
    dragStartSignal = {};
    dragEndSignal = {};
    dragSignal = {};
    cursorEntranceSignal = {};
    charPressedSignal = {};
    keyUpSignal = {};
    keyDownSignal = {};
    keyRepeatSignal = {};
    spaceMouseMoveSignal = {};
    spaceMouseDownSignal = {};
    spaceMouseUpSignal = {};
    spaceMouseRepeatSignal = {};
    preDrawSignal = {};
    preDrawPostViewportSignal = {};
    drawSignal = {};
    postDrawPreViewportSignal = {};
    postDrawSignal = {};
    objectsLoadedSignal = {};
    dragDropSignal = {};
    postResizeSignal = {};
    postRescaleSignal = {};
    interruptCloseSignal = {};
    touchStartSignal = {};
    touchMoveSignal = {};
    touchEndSignal = {};
    touchpadRotateGestureBeginSignal = {};
    touchpadRotateGestureUpdateSignal = {};
    touchpadRotateGestureEndSignal = {};
    touchpadSwipeGestureBeginSignal = {};
    touchpadSwipeGestureUpdateSignal = {};
    touchpadSwipeGestureEndSignal = {};
    touchpadZoomGestureBeginSignal = {};
    touchpadZoomGestureUpdateSignal = {};
    touchpadZoomGestureEndSignal = {};
    postFocusSignal = {};
}

void Viewer::init_()
{
    MR_TIMER;
    initBasisAxesObject_();
    initBasisViewControllerObject_();
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
        auto & pl = *plugins[i];
        spdlog::debug( "Shutting down plugin: {}", typeid( pl ).name() );
        pl.shutdown();
    }
    spdlog::debug( "All plugins shut down" );
    if ( menuPlugin_ )
    {
        auto & pl = *menuPlugin_;
        spdlog::debug( "Shutting down menu plugin: {}", typeid( pl ).name() );
        pl.shutdown();
        spdlog::debug( "Menu plugin shut down" );
    }
}

void Viewer::postEmptyEvent()
{
    if ( !isGLInitialized() )
        return;
#ifdef __EMSCRIPTEN__
    emplaceEvent( "Empty", [] () {} );
#endif
    glfwPostEmptyEvent();
}

const TouchpadParameters & Viewer::getTouchpadParameters() const
{
    if ( !touchpadController_ )
    {
        const static TouchpadParameters empty;
        return empty;
    }
    return touchpadController_->getParameters();
}

void Viewer::setTouchpadParameters( const TouchpadParameters & ps )
{
    if ( !touchpadController_ )
        touchpadController_ = std::make_unique<TouchpadController>();
    touchpadController_->setParameters( ps );
}

SpaceMouseParameters Viewer::getSpaceMouseParameters() const
{
    if ( !spaceMouseController_ )
        return {};
    return spaceMouseController_->getParameters();
}

void Viewer::setSpaceMouseParameters( const SpaceMouseParameters & ps )
{
    if ( !spaceMouseController_ )
        spaceMouseController_ = std::make_unique<SpaceMouseController>();
    spaceMouseController_->setParameters( ps );
}

Viewer::Viewer() :
    selected_viewport_index( 0 ),
    eventQueue_( std::make_unique<ViewerEventQueue>() ),
    mouseController_( std::make_unique<MouseController>() ),
    recentFilesStore_( std::make_unique<RecentFilesStore>() ),
    frameCounter_( std::make_unique<FrameCounter>() )
{
    window = nullptr;

    viewport_list.reserve( 32 );
    viewport_list.emplace_back();
    viewport_list.front().id = ViewportId{ 1 };
    presentViewportsMask_ |= viewport_list.front().id;

    resetSettingsFunction = [] ( Viewer* viewer )
    {
        viewer->glPickRadius = 0;
        viewer->scrollForce = 1.0f;
        viewer->experimentalFeatures = false;
        viewer->setSpaceMouseParameters( SpaceMouseParameters{} );
        viewer->setTouchpadParameters( TouchpadParameters{} );
        viewer->enableAlphaSort( true );

        for ( ViewportId id : viewer->getPresentViewports() )
        {
            Viewport& viewport = viewer->viewport( id );
            // Reset selected parameters
            Viewport::Parameters defaultParams;
            Viewport::Parameters params = viewport.getParameters();
            params.cameraZoom = defaultParams.cameraZoom;
            params.cameraViewAngle = defaultParams.cameraViewAngle;
            params.cameraDnear = defaultParams.cameraDnear;
            params.cameraDfar = defaultParams.cameraDfar;
            params.depthTest = defaultParams.depthTest;
            params.orthographic = defaultParams.orthographic;
            params.borderColor = defaultParams.borderColor;
            params.clippingPlane = defaultParams.clippingPlane;
            params.rotationMode = defaultParams.rotationMode;
            viewport.setParameters( params );
            // Reset other properties
            viewer->viewport().showAxes( true );
            viewer->viewport().showGlobalBasis( false );
            viewer->viewport().showRotationCenter( true );
            viewer->viewport().showClippingPlane( false );
        }
    };
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
    if ( !std::filesystem::exists( mesh_file_name, ec ) )
        return false;
    if ( !std::filesystem::is_regular_file( mesh_file_name, ec ) )
        return false;

    std::string ext = utf8string( mesh_file_name.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    for ( auto& filter : MeshLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : LinesLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : PointsLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : DistanceMapLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : SceneLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : ObjectLoad::getFilters() )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }
    for ( auto& filter : GcodeLoad::Filters )
    {
        if ( filter.extensions.find( ext ) != std::string::npos )
            return true;
    }

    return false;
}

/// returns class name of all objects or nullopt if they are of different types
static std::optional<std::string> commonClassName( const std::vector<std::shared_ptr<Object>>& objs )
{
    if ( objs.empty() )
        return {};

    auto cn = objs[0]->className();
    if ( objs.size() == 1 )
        return cn;

    for ( int i = 1; i < objs.size(); ++i )
        if ( cn != objs[i]->className() )
            return {};

    return objs[0]->classNameInPlural();
}

bool Viewer::loadFiles( const std::vector<std::filesystem::path>& filesList, const FileLoadOptions & options )
{
    if ( filesList.empty() )
        return false;

    const auto postProcess = [this, options] ( const SceneLoad::SceneLoadResult& result )
    {
        if ( result.scene )
        {
            const bool wasEmptyScene = SceneRoot::get().children().empty();
            const bool wasEmptyUndo = globalHistoryStore_ && globalHistoryStore_->getStackPointer() == 0;

            std::string undoName = options.undoPrefix + commonFilesName( result.loadedFiles );
            if ( auto cn = commonClassName( result.scene->children() ) )
                undoName += " as " + *cn;

            bool singleSceneFile = result.loadedFiles.size() == 1 && !result.isSceneConstructed;
            bool forceReplace = options.replaceMode == FileLoadOptions::ReplaceMode::ForceReplace;
            bool forceAdd = options.replaceMode == FileLoadOptions::ReplaceMode::ForceAdd;

            if ( forceReplace || wasEmptyScene || ( !forceAdd && singleSceneFile ) )
            {
                {
                    // the scene is taken as is from a single file, replace the current scene with it
                    AppendHistory<SwapRootAction>( undoName );
                    auto newRoot = result.scene;
                    std::swap( newRoot, SceneRoot::getSharedPtr() );
                    setSceneDirty();
                    onSceneSaved( result.loadedFiles.front() );
                }
                if ( options.loadedCallback ) // strictly after history is added
                    options.loadedCallback( SceneRoot::get().children(), result.errorSummary, result.warningSummary );
                objectsLoadedSignal( SceneRoot::get().children(), result.errorSummary, result.warningSummary );
            }
            else
            {
                // not-scene file was open, or several scenes were open, append them to the current scene
                for ( const auto& file : result.loadedFiles )
                    recentFilesStore().storeFile( file );
                const auto children = result.scene->children();
                {
                    SCOPED_HISTORY( undoName );

                    result.scene->removeAllChildren();
                    for ( const auto& obj : children )
                    {
                        AppendHistory<ChangeSceneAction>( "add obj", obj, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( obj );
                    }
                }
                if ( options.loadedCallback ) // strictly after history is added
                    options.loadedCallback( children, result.errorSummary, result.warningSummary );
                objectsLoadedSignal( children, result.errorSummary, result.warningSummary );
            }

            // if the original state was empty, avoid user confusion when they undo opening and see empty modified scene
            if ( wasEmptyScene && wasEmptyUndo && globalHistoryStore_ )
            {
                globalHistoryStore_->clear();
                globalHistoryStore_->setSavedState();
                makeTitleFromSceneRootPath();
            }

            viewport().preciseFitDataToScreenBorder( { 0.9f } );
        }
        else
        {
            if ( options.loadedCallback )
                options.loadedCallback( {}, result.errorSummary, result.warningSummary );
            objectsLoadedSignal( {}, result.errorSummary, result.warningSummary );
        }
        if ( !result.errorSummary.empty() )
            showModal( result.errorSummary, NotificationType::Error );
        else if ( !result.warningSummary.empty() )
        {
            NotificationTagMask mask = NotificationTags::None;
            if ( result.warningSummary.find( "consider using" ) != std::string::npos )
                mask |= NotificationTags::Recommendation;
            if ( result.warningSummary.find( "were duplicated" ) != std::string::npos ||
                result.warningSummary.find( "were skipped" ) != std::string::npos )
                mask |= NotificationTags::ImplicitChanges;
            if ( mask == NotificationTags::None )
                mask = NotificationTags::ImplicitChanges;
            pushNotification( { .text = result.warningSummary, .type = NotificationType::Warning,.tags = mask } );
        }
    };

#if defined( __EMSCRIPTEN__ ) && !defined( __EMSCRIPTEN_PTHREADS__ )
    ProgressBar::orderWithManualFinish( "Open files", [filesList, postProcess]
    {
        SceneLoad::asyncFromAnySupportedFormat( filesList, [postProcess] ( SceneLoad::SceneLoadResult result )
        {
            postProcess( result );
            ProgressBar::finish();
        }, ProgressBar::callBackSetProgress );
    } );
#else
    ProgressBar::orderWithMainThreadPostProcessing( "Open files", [filesList, postProcess]
    {
        auto result = SceneLoad::fromAnySupportedFormat( filesList, ProgressBar::callBackSetProgress );
        return [result = std::move( result ), postProcess]
        {
            postProcess( result );
        };
    } );
#endif

    MR::FileDialog::setLastUsedDir( utf8string( filesList[0].parent_path() ) );
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

bool Viewer::mouseClick( MouseButton button, int modifier )
{
    return mouseClickSignal( button, modifier );
}

bool Viewer::dragStart( MouseButton button, int modifier )
{
    return dragStartSignal( button, modifier );
}

bool Viewer::dragEnd( MouseButton button, int modifier )
{
    return dragEndSignal( button, modifier );
}

bool Viewer::drag( int mouse_x, int mouse_y )
{
    return dragSignal( mouse_x, mouse_y );
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
    MR_TIMER;
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
    spdlog::info( window ? "glfwCreateWindow succeeded" : "glfwCreateWindow failed" );
    return bool( window );
}

bool Viewer::needRedraw_() const
{
    if ( dirtyScene_ )
        return true;

    for ( const auto& viewport : viewport_list )
        if ( viewport.getRedrawFlag() )
            return true;

    if ( globalBasis && globalBasis->getRedrawFlag( presentViewportsMask_ ) )
        return true;

    if ( basisAxes && basisAxes->getRedrawFlag( presentViewportsMask_ ) )
        return true;

    if ( basisViewController && basisViewController->getRedrawFlag( presentViewportsMask_ ) )
        return true;

    return getRedrawFlagRecursive( SceneRoot::get(), presentViewportsMask_ );
}

void Viewer::resetRedraw_()
{
    dirtyScene_ = false;

    for ( auto& viewport : viewport_list )
        viewport.resetRedrawFlag();

    if ( globalBasis )
        globalBasis->resetRedrawFlag();

    if ( basisAxes )
        basisAxes->resetRedrawFlag();

    if ( basisViewController )
        basisViewController->resetRedrawFlag();

    resetRedrawFlagRecursive( SceneRoot::get() );
}

void Viewer::recursiveDraw_( const Viewport& vp, const Object& obj, const AffineXf3f& parentXf, RenderModelPassMask renderType, int* numDraws ) const
{
    if ( !obj.isVisible( vp.id ) )
        return;
    auto xfCopy = parentXf * obj.xf( vp.id );
    auto visObj = obj.asType<VisualObject>();
    if ( visObj )
    {
        if ( vp.draw( *visObj, xfCopy, DepthFunction::Default, renderType, alphaSortEnabled_ ) )
        {
            if ( numDraws )
                ++( *numDraws );
        }
    }
    for ( const auto& child : obj.children() )
        recursiveDraw_( vp, *child, xfCopy, renderType, numDraws );
}

void Viewer::draw( bool force )
{
#ifdef __EMSCRIPTEN__
    (void)force;
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
    SceneCache::invalidateAll();
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

    frameCounter_->startDraw();

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
    frameCounter_->endDraw( swapped );
    isInDraw_ = false;
    return ( window && swapped );
}

void Viewer::drawUiRenderObjects_()
{
    // Currently, a part of the contract of `IRenderObject::renderUi()` is that at most rendering task is in flight at any given time.
    // That's why each viewport is being drawn separately.
    if ( !window )
        return;
    UiRenderManager& uiRenderManager = getMenuPlugin()->getUiRenderManager();

    for ( Viewport& viewport : getViewerInstance().viewport_list )
    {
        UiRenderParams renderParams{ viewport.getBaseRenderParams() };
        renderParams.scale = menuPlugin_->menu_scaling();

        uiRenderManager.preRenderViewport( viewport.id );
        MR_FINALLY{ uiRenderManager.postRenderViewport( viewport.id ); };

        UiRenderParams::UiTaskList tasks;
        tasks.reserve( 50 );
        renderParams.tasks = &tasks;

        auto lambda = [&]( auto& lambda, Object& object ) -> void
        {
            if ( !object.isVisible( viewport.id ) )
                return;

            if ( auto visual = dynamic_cast<VisualObject*>( &object ) )
                visual->renderUi( renderParams );

            for ( const auto& child : object.children() )
                lambda( lambda, *child );
        };
        lambda( lambda, SceneRoot::get() );

        auto backwardPassParams = uiRenderManager.beginBackwardPass( viewport.id, tasks );

        std::sort( tasks.begin(), tasks.end(), []( const auto& a, const auto& b ){ return a->renderTaskDepth > b->renderTaskDepth; } );
        for ( auto it = tasks.end(); it != tasks.begin(); )
        {
            --it;
            ( *it )->earlyBackwardPass( backwardPassParams );
        }
        uiRenderManager.finishBackwardPass( backwardPassParams );

        for ( const auto& task : tasks )
            task->renderPass();
    }
}

void Viewer::drawFull( bool dirtyScene )
{
    // unbind to clean main framebuffer
    if ( sceneTexture_ )
        sceneTexture_->unbind();
    // clean main framebuffer
    clearFramebuffers();

    if ( menuPlugin_ )
        menuPlugin_->startFrame();

    if ( sceneTexture_ )
    {
        sceneTexture_->bind( true );
        // need to clean it in texture too
        clearFramebuffers();
    }
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
    {
        drawUiRenderObjects_();
        menuPlugin_->finishFrame();
    }
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
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), RenderModelPassMask::Opaque );
#ifndef __EMSCRIPTEN__
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), RenderModelPassMask::VolumeRendering );
#endif
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), RenderModelPassMask::Transparent, &numTransparent );
    }

    drawSignal();

    if ( numTransparent > 0 && alphaSortEnabled_ )
    {
        alphaSorter_->drawTransparencyTextureToScreen();
        alphaSorter_->clearTransparencyTextures();
    }
    // draw after alpha texture
    for ( const auto& viewport : viewport_list )
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), RenderModelPassMask::NoDepthTest );

    postDrawPreViewportSignal();

    for ( const auto& viewport : viewport_list )
        viewport.postDraw();

    resetRedraw_();
}

void Viewer::setupScene()
{
    preSetupViewSignal();
    for ( auto& viewport : viewport_list )
        viewport.setupView();
}

void Viewer::clearFramebuffers()
{
    for ( auto& viewport : viewport_list )
        viewport.clearFramebuffers();
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
        sceneTexture_->reset( framebufferSize, getMSAAPow( getRequiredMSAA_( true, true ) ) );

#if !defined(__EMSCRIPTEN__) || defined(MR_EMSCRIPTEN_ASYNCIFY)
    if ( isLaunched_ && !isInDraw_ )
    {
        incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, true );
        while ( !draw_( true ) );
    }
#endif

    if ( hasScaledFramebuffer_ )
        updatePixelRatio_();

    gWindowSizeInitialized = true;
}

void Viewer::postSetPosition( int xPos, int yPos )
{
    if ( !windowMaximized && !glfwGetWindowMonitor( window ) )
        windowSavePos = { xPos, yPos };

    if ( hasScaledFramebuffer_ )
        updatePixelRatio_();
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

void Viewer::postClose()
{
    incrementForceRedrawFrames( forceRedrawMinimumIncrementAfterEvents, swapOnLastPostEventsRedraw );
    postEmptyEvent();
#ifndef __EMSCRIPTEN__
    if ( window )
        glfwRequestWindowAttention( window );
#endif
}

void Viewer::set_root( SceneRootObject& newRoot )
{
    std::swap( SceneRoot::get(), newRoot );
}

void Viewer::initGlobalBasisAxesObject_()
{
    globalBasis = std::make_unique<ViewportGlobalBasis>();
}

void Viewer::initBasisAxesObject_()
{
    // store basis axes in the corner
    const float cubeSzie = 0.8f;
    const float size = 1.0f;
    std::shared_ptr<Mesh> basisAxesMesh = std::make_shared<Mesh>( makeBasisAxes( size, size * 0.03f, size * 0.03f, 0.0f ) );
    const Vector3f translation = Vector3f::diagonal( -cubeSzie * 0.5f );
    basisAxesMesh->transform( AffineXf3f::translation( translation ) );

    basisAxes = std::make_shared<ObjectMesh>();
    basisAxes->setMesh( basisAxesMesh );
    basisAxes->setName("Basis axes mesh");
    basisAxes->setFlatShading( true );

    auto numF = basisAxesMesh->topology.edgePerFace().size();
    // setting color to faces
    const Color colorX = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisX );
    const Color colorY = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisY );
    const Color colorZ = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisZ );
    FaceColors colorMap( numF );
    const auto arrowSize = numF / 3;
    for (int i = 0; i < arrowSize; i++)
    {
        colorMap[FaceId( i )] = colorX;
        colorMap[FaceId( i + arrowSize )] = colorY;
        colorMap[FaceId( i + arrowSize * 2 )] = colorZ;
    }
    const float labelPos = size + 0.2f;

    addLabel( *basisAxes, "X", labelPos * Vector3f::plusX() + translation, true );
    addLabel( *basisAxes, "Y", labelPos * Vector3f::plusY() + translation, true );
    addLabel( *basisAxes, "Z", labelPos * Vector3f::plusZ() + translation, true );

    basisAxes->setFacesColorMap( std::move( colorMap ) );
    basisAxes->setVisualizeProperty( false, MeshVisualizePropertyType::EnableShading, ViewportMask::all() );
    basisAxes->setColoringType( ColoringType::FacesColorMap );

    uiUpdateConnections_.push_back( ColorTheme::instance().onChanged( [this, numF] ()
    {
        if ( !basisAxes )
            return;

        const Color colorX = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisX );
        const Color colorY = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisY );
        const Color colorZ = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisZ );
        FaceColors colorMap;
        basisAxes->updateFacesColorMap( colorMap ); // swap with empty with real
        const auto arrowSize = numF / 3;
        for ( int i = 0; i < arrowSize; i++ ) // update real
        {
            colorMap[FaceId( i )] = colorX;
            colorMap[FaceId( i + arrowSize )] = colorY;
            colorMap[FaceId( i + arrowSize * 2 )] = colorZ;
        }
        basisAxes->updateFacesColorMap( colorMap ); // swap updated real with empty

        const Color& color = SceneColors::get( SceneColors::Type::Labels );

        auto labels = getAllObjectsInTree<ObjectLabel>( basisAxes.get(), ObjectSelectivityType::Any );
        for ( const auto& label : labels )
        {
            label->setFrontColor( color, true );
            label->setFrontColor( color, false );
        }
    } ) );
    uiUpdateConnections_.push_back( postRescaleSignal.connect( [this] ( float, float )
    {
        if ( !menuPlugin_ )
            return;
        auto labels = getAllObjectsInTree<ObjectLabel>( basisAxes.get(), ObjectSelectivityType::Any );
        for ( const auto& label : labels )
            label->setFontHeight( 20.0f * menuPlugin_->menu_scaling() );
    } ) );
}

void Viewer::initBasisViewControllerObject_()
{
    basisViewController = std::make_unique<CornerControllerObject>();
    basisViewController->initDefault();
}

void Viewer::initClippingPlaneObject_()
{
    std::shared_ptr<Mesh> plane = std::make_shared<Mesh>( makePlane() );
    clippingPlaneObject = std::make_shared<ObjectMesh>();
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
    rotationSphere = std::make_shared<ObjectMesh>();
    rotationSphere->setFrontColor( color, false );
    rotationSphere->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    rotationSphere->setAncillary( true );
}

void Viewer::initSpaceMouseHandler( [[maybe_unused]] std::function<void(const std::string&)> deviceSignal )
{
    spaceMouseHandler_.reset();
#ifndef __EMSCRIPTEN__
#ifdef __APPLE__
    // try to use the official driver first
    auto driverHandler = std::make_unique<SpaceMouseHandler3dxMacDriver>();
    driverHandler->setClientName( MR_PROJECT_NAME );
    if ( driverHandler->initialize( deviceSignal ) )
    {
        spaceMouseHandler_ = std::move( driverHandler );
        spdlog::info( "SpaceMouseHandler3dxMacDriver initialized" );
        return;
    }

    // fallback to the HIDAPI implementation
    spdlog::warn( "Failed to find or use the 3DxWare driver; falling back to the HIDAPI implementation" );
#endif
    spaceMouseHandler_ = std::make_unique<SpaceMouseHandlerHidapi>();
    if ( spaceMouseHandler_->initialize( std::move( deviceSignal ) ) )
    {
        spdlog::info( "SpaceMouseHandlerHidapi initialized" );
    }
    else
    {
        spdlog::warn( "Failed to initialize SpaceMouseHandlerHidapi" );
    }
#endif
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

    if ( windowTitle )
        windowTitle->setSceneName( sceneFileName );
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

ViewportId Viewer::append_viewport( const ViewportRectangle & viewportRect, bool append_empty /*= false */ )
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
    const auto& currentPos = mouseController_->getMousePos();
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

void Viewer::preciseFitDataViewport( MR::ViewportMask vpList, const FitDataParams& params )
{
    for( auto& viewport : viewport_list )
    {
        if( viewport.id.value() & vpList.value() )
        {
            viewport.preciseFitDataToScreenBorder( params );
        }
    }
}

void Viewer::preciseFitDataViewport( MR::ViewportMask vpList )
{
    return preciseFitDataViewport( vpList, {} );
}

size_t Viewer::getTotalFrames() const
{
    return frameCounter_->totalFrameCounter;
}

size_t Viewer::getSwappedFrames() const
{
    return frameCounter_->swappedFrameCounter;
}

size_t Viewer::getFPS() const
{
    return frameCounter_->fps;
}

double Viewer::getPrevFrameDrawTimeMillisec() const
{
    return frameCounter_->drawTimeMilliSec.count();
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
    frameCounter_->reset();
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
    clearFramebuffers();
    drawScene();

    fd.copyTextureBindDef();
    fd.bindTexture();

#ifdef __EMSCRIPTEN__
    GLuint fbo;
    GL_EXEC( glGenFramebuffers(1, &fbo) );
    GL_EXEC( glBindFramebuffer(GL_FRAMEBUFFER, fbo) );
    GL_EXEC( glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fd.getTexture(), 0) );

    GL_EXEC( glReadPixels(0, 0, newRes.x, newRes.y, GL_RGBA, GL_UNSIGNED_BYTE, ( void* )( pixels.data() )) );

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
#else
    GL_EXEC( glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ( void* )( pixels.data() ) ) );
#endif

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

bool Viewer::isSceneTextureBound() const
{
    if ( !sceneTexture_ )
        return false;
    return sceneTexture_->isBound();
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

bool Viewer::isSceneTextureEnabled() const
{
    return bool( sceneTexture_ );
}

int Viewer::getMSAA() const
{
    int curSamples = 0;
    if ( !glInitialized_ )
        return curSamples;
    if ( !sceneTexture_ || sceneTexture_->isBound() )
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &curSamples ) );
    else
    {
        sceneTexture_->bind( false );
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &curSamples ) );
        sceneTexture_->unbind();
    }
    return curSamples;
}

void Viewer::requestChangeMSAA( int newMSAA )
{
    if ( settingsMng_ )
        settingsMng_->saveInt( "multisampleAntiAliasing", newMSAA );
    if ( sceneTexture_ )
    {
        CommandLoop::appendCommand( [newMSAA, this] ()
        {
            sceneTexture_->reset( framebufferSize, getMSAAPow( newMSAA ) );
            setSceneDirty();
        } );
    }
}

int Viewer::getRequestedMSAA() const
{
    if ( sceneTexture_ )
        return getRequiredMSAA_( true, true );
    return getRequiredMSAA_( false, false );
}

void Viewer::setViewportSettingsManager( std::unique_ptr<IViewerSettingsManager> mng )
{
    settingsMng_ = std::move( mng );
}

PointInAllSpaces Viewer::getMousePointInfo() const
{
    const auto& currentPos = mouseController_->getMousePos();
    return getPixelPointInfo( Vector3f( float( currentPos.x ), float( currentPos.y ), 0.f ) );
}

PointInAllSpaces Viewer::getPixelPointInfo( const Vector3f& screenPoint ) const
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
            auto [obj, pick] = viewport.pickRenderObject( { .point = Vector2f( res.viewportSpace.x, res.viewportSpace.y ) } );
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
    {
        globalHistoryStore_ = std::make_shared<HistoryStore>();
        globalHistoryStore_->changedSignal.connect( [this]( const HistoryStore&, HistoryStore::ChangeType type )
        {
            if ( type == HistoryStore::ChangeType::PostUndo ||
                 type == HistoryStore::ChangeType::PostRedo ||
                 type == HistoryStore::ChangeType::AppendAction )
                makeTitleFromSceneRootPath();
        } );
    }
    else
        globalHistoryStore_.reset();
}

void Viewer::appendHistoryAction( const std::shared_ptr<HistoryAction>& action )
{
    if ( globalHistoryStore_ )
        globalHistoryStore_->appendAction( action );
}

bool Viewer::globalHistoryUndo()
{
    return globalHistoryStore_ && globalHistoryStore_->undo();
}

bool Viewer::globalHistoryRedo()
{
    return globalHistoryStore_ && globalHistoryStore_->redo();
}

void Viewer::onSceneSaved( const std::filesystem::path& savePath, bool storeInRecent )
{
    if ( !savePath.empty() && storeInRecent )
        recentFilesStore().storeFile( savePath );

    SceneRoot::setScenePath( savePath );

    if ( globalHistoryStore_ )
        globalHistoryStore_->setSavedState();

    makeTitleFromSceneRootPath();
}

const std::shared_ptr<ImGuiMenu>& Viewer::getMenuPlugin() const
{
    return menuPlugin_;
}

std::shared_ptr<RibbonMenu> Viewer::getRibbonMenu() const
{
    return std::dynamic_pointer_cast<RibbonMenu>( getMenuPlugin() );
}

void Viewer::setMenuPlugin( std::shared_ptr<ImGuiMenu> menu )
{
    assert( !menuPlugin_ );
    assert( menu );
    menuPlugin_ = menu;
}

void Viewer::stopEventLoop()
{
    stopEventLoop_ = true;
    postClose();
}

size_t Viewer::getStaticGLBufferSize() const
{
    return GLStaticHolder::getStaticGLBuffer().heapBytes();
}

void Viewer::updatePixelRatio_()
{
    int winWidth, winHeight;
    glfwGetWindowSize( window, &winWidth, &winHeight );
    pixelRatio = float( framebufferSize.x ) / float( winWidth );
}

int Viewer::getRequiredMSAA_( bool sceneTextureOn, bool forSceneTexture ) const
{
    if ( !sceneTextureOn && forSceneTexture )
    {
        assert( false );
        return -1;
    }
    if ( sceneTextureOn && !forSceneTexture )
        return 1; // disable msaa for main framebuffer if scene texture is used
    
    int cDefaultMSAA = 8;
#if defined(__EMSCRIPTEN__)
    cDefaultMSAA = 4;
    if ( bool( EM_ASM_INT( return is_mac() ) ) && forSceneTexture )
        cDefaultMSAA = 2;
#elif defined(__APPLE__)
    cDefaultMSAA = 2;
#endif
    if ( !settingsMng_ )
        return cDefaultMSAA;
    return settingsMng_->loadInt( "multisampleAntiAliasing", cDefaultMSAA );
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
