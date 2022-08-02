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
#include "MRShadersHolder.h"
#include "MRViewerPlugin.h"
#include "MRCommandLoop.h"
#include "MRSplashWindow.h"
#include "MRViewerSettingsManager.h"
#include "MRGladGlfw.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRPch/MRWasm.h"

#ifdef __EMSCRIPTEN__
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
        MR::getViewerInstance().post_rescale( float( sEmsPixelRatio ), float( sEmsPixelRatio ) );
    }
    float newWidth = width * pixelRatio;
    float newHeight = height * pixelRatio;
    glfwSetWindowSize( MR::getViewerInstance().window, int( newWidth ), int( newHeight ) );
    return 1;
}

}
#endif
#include "MRMesh/MRSerializer.h"

static void glfw_mouse_press( GLFWwindow* /*window*/, int button, int action, int modifier )
{
    MR::Viewer::MouseButton mb;

    if ( button == GLFW_MOUSE_BUTTON_1 )
        mb = MR::Viewer::MouseButton::Left;
    else if ( button == GLFW_MOUSE_BUTTON_2 )
        mb = MR::Viewer::MouseButton::Right;
    else //if (button == GLFW_MOUSE_BUTTON_3)
        mb = MR::Viewer::MouseButton::Middle;

    if ( action == GLFW_PRESS )
        MR::Viewer::instanceRef().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Down, [mb, modifier] ()
    {
        MR::Viewer::instanceRef().mouse_down( mb, modifier );
    } );        
    else
        MR::Viewer::instanceRef().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Up, [mb, modifier] ()
    {
        MR::Viewer::instanceRef().mouse_up( mb, modifier );
    } );
}

static void glfw_error_callback( int /*error*/, const char* description )
{
    spdlog::error( description );
}

static void glfw_char_mods_callback( GLFWwindow* /*window*/, unsigned int codepoint )
{
    MR::Viewer::instanceRef().key_pressed( codepoint, 0 );
}

static void glfw_key_callback( GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int modifier )
{
    if ( action == GLFW_PRESS )
        MR::Viewer::instanceRef().key_down( key, modifier );
    else if ( action == GLFW_RELEASE )
        MR::Viewer::instanceRef().key_up( key, modifier );
    else if ( action == GLFW_REPEAT )
        MR::Viewer::instanceRef().key_repeat( key, modifier );
}

static void glfw_window_size( GLFWwindow* /*window*/, int width, int height )
{
    MR::Viewer::instanceRef().post_resize( width, height );
}

static void glfw_window_pos( GLFWwindow* /*window*/, int xPos, int yPos )
{
    // need for remember window pos and size before maximize
    // located here because glfw_window_pos calls before glfw_window_maximize and glfw_window_iconify (experience)
    auto& viewer = MR::Viewer::instanceRef();
    viewer.windowOldPos = viewer.windowSavePos;
    viewer.postSetPosition( xPos, yPos );
}

#ifndef __EMSCRIPTEN__
static void glfw_window_maximize( GLFWwindow* /*window*/, int maximized )
{
    MR::Viewer::instanceRef().postSetMaximized( bool( maximized ) );
}

static void glfw_window_iconify( GLFWwindow* /*window*/, int iconified )
{
    MR::Viewer::instanceRef().postSetIconified( bool( iconified ) );
}

static void glfw_window_focus( GLFWwindow* /*window*/, int focused )
{
    MR::Viewer::instanceRef().postFocus( bool( focused ) );
}
#endif

static void glfw_window_scale( GLFWwindow* /*window*/, float xscale, float yscale )
{
    MR::Viewer::instanceRef().post_rescale( xscale, yscale );
}

static void glfw_mouse_move( GLFWwindow* /*window*/, double x, double y )
{
    auto eventCall = [x, y] ()
    {
        MR::Viewer::instanceRef().mouse_move( int( x ), int( y ) );
        MR::Viewer::instanceRef().draw();
    };
    if ( MR::Viewer::instanceRef().mouseEventQueue.empty() || 
         MR::Viewer::instanceRef().mouseEventQueue.back().type != MR::Viewer::MouseQueueEvent::Type::Move )
    {
        MR::Viewer::instanceRef().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Move, std::move( eventCall ) );
    }
    else
    {
        // if last event in this frame was move - replace it with newer one
        MR::Viewer::instanceRef().mouseEventQueue.back().callEvent = std::move( eventCall );
    }
}

static void glfw_mouse_scroll( GLFWwindow* /*window*/, double /*x*/, double y )
{
    MR::Viewer::instanceRef().mouse_scroll( float( y ) );
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
    MR::Viewer::instanceRef().mouseEventQueue.emplace( MR::Viewer::MouseQueueEvent::Type::Drop, [paths] ()
    {
        MR::Viewer::instanceRef().drag_drop( paths );
    } );
}

namespace MR
{

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
    setup.setupExtendedLibraries();

    return viewer.launch( params );
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
        else if ( flag == "-width" )
            nextW = true;
        else if ( flag == "-height" )
            nextH = true;
    }
}

#ifdef __EMSCRIPTEN__
void Viewer::mainLoopFunc_()
{
    instanceRef().draw(true);
    instanceRef().processMouseEventsQueue_();
    CommandLoop::processCommands();
}
#endif

int Viewer::launch( const LaunchParams& params )
{
    if ( isLaunched_ )
    {
        spdlog::error( "Viewer is already launched!" );
        return 1;
    }
    isAnimating = params.isAnimating;
    animationMaxFps = params.animationMaxFps;
    auto res = launchInit_( params );
    if ( res != EXIT_SUCCESS )
        return res;

    CommandLoop::processCommands(); // execute pre init commands before first draw
    focusRedrawReady_ = true;

    if ( params.windowMode == LaunchParams::HideInit && window )
        glfwShowWindow( window );

    parseCommandLine_( params.argc, params.argv );

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(mainLoopFunc_, 0, true);
#else
    if ( params.startEventLoop )
        launchEventLoop();
    if ( params.close )
        launchShut();
#endif
    return EXIT_SUCCESS;
}

bool Viewer::checkOpenGL_(const LaunchParams& params )
{
    int windowWidth = params.width;
    int windowHeight = params.height;
#ifdef __APPLE__
    alphaSorter_.reset();
    spdlog::warn( "Alpha sort is not available" );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    spdlog::warn( "Loading OpenGL 3.3 for macOS" );
    if ( !tryCreateWindow_( params.fullscreen, windowWidth, windowHeight, params.name, 3, 3 ) )
    {
        spdlog::critical( "Cannot load OpenGL 3.3" );
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
        return EXIT_FAILURE;
    }
#if defined(__APPLE__)
    //Setting window properties
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE );
#endif
    glfwWindowHint( GLFW_SAMPLES, 8 );
#ifndef __EMSCRIPTEN__
    glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, params.enableTransparentBackground );
#endif

    alphaSorter_ = std::make_unique<AlphaSortGL>();

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
#ifndef NDEBUG
#ifndef __EMSCRIPTEN__
        spdlog::debug( "OpenGL Version {}.{} loaded", GLVersion.major, GLVersion.minor );
#endif
        int major, minor, rev;
        major = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MAJOR );
        minor = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MINOR );
        rev = glfwGetWindowAttrib( window, GLFW_CONTEXT_REVISION );
        spdlog::debug( "OpenGL version received: {}.{}.{}", major, minor, rev );
        if ( glInitialized_ )
        {
            spdlog::debug( "Supported OpenGL is {}", ( const char* )glGetString( GL_VERSION ) );
            spdlog::debug( "Supported GLSL is {}", ( const char* )glGetString( GL_SHADING_LANGUAGE_VERSION ) );
        }
#endif
        if ( params.showMRVersionInTitle )
        {
            defaultWindowTitle = params.name + " (" + GetMRVersionString() + ")";
            glfwSetWindowTitle( window, defaultWindowTitle.c_str() );
        }

        glfwSetInputMode( window, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
        // Register callbacks
        glfwSetKeyCallback( window, glfw_key_callback );
        glfwSetCursorPosCallback( window, glfw_mouse_move );
        glfwSetWindowSizeCallback( window, glfw_window_size );
        glfwSetWindowPosCallback( window, glfw_window_pos );
#ifndef __EMSCRIPTEN__
        glfwSetWindowMaximizeCallback( window, glfw_window_maximize );
        glfwSetWindowIconifyCallback( window, glfw_window_iconify );
        glfwSetWindowContentScaleCallback( window, glfw_window_scale );
        glfwSetWindowFocusCallback( window, glfw_window_focus );
#endif
        glfwSetMouseButtonCallback( window, glfw_mouse_press );
        glfwSetCharCallback( window, glfw_char_mods_callback );
        glfwSetDropCallback( window, glfw_drop_callback );
        // Handle retina displays (windows and mac)
        int width, height;
        glfwGetFramebufferSize( window, &width, &height );
        int width_window, height_window;
        glfwGetWindowSize( window, &width_window, &height_window );
        // Initialize IGL viewer
        glfw_window_size( window, width_window, height_window );

        float xscale{ 1.0f }, yscale{ 1.0f };
#ifndef __EMSCRIPTEN__
        glfwGetWindowContentScale( window, &xscale, &yscale );
#endif
        glfw_window_scale( window, xscale, yscale );

        enableAlphaSort( true );

        if ( alphaSorter_ )
        {
            alphaSorter_->init();
            alphaSorter_->updateTransparencyTexturesSize( width_window, height_window );
        }

        mouseController.connect();
    }

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

    if ( menuPlugin_ )
        menuPlugin_->init( this );
    init_();
    // it is replaced here because some plugins can rise modal window, and scroll event sometimes can pass over it
    if ( window )
        glfwSetScrollCallback( window, glfw_mouse_scroll );
    // give it name of app to store in right place
    recentFilesStore = RecentFilesStore( params.name );

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
            processMouseEventsQueue_();
            CommandLoop::processCommands();
        } while ( ( !( window && glfwWindowShouldClose( window ) ) && !stopEventLoop_ ) && ( forceRedrawFrames_ > 0 || needRedraw_() ) );

        if ( isAnimating )
        {
            const double minDuration = 1.0 / double( animationMaxFps );
            glfwWaitEventsTimeout( minDuration );
            processMouseEventsQueue_();
        }
        else
        {
            glfwWaitEvents();
            processMouseEventsQueue_();
        }
    }
}

void Viewer::launchShut()
{
    if ( !isLaunched_ )
    {
        spdlog::error( "Viewer is not launched!" );
        return;
    }

    if ( settingsMng_ )
        settingsMng_->saveSettings( *this );

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

    ShadersHolder::freeAllShaders();

    alphaSorter_.reset();

    glfwDestroyWindow( window );
    glfwTerminate();
    glInitialized_ = false;
    isLaunched_ = false;
    return;
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
        settingsMng_->loadSettings( *this );
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

void Viewer::parseCommandLine_( int argc, char** argv )
{
#ifndef __EMSCRIPTEN__
    bool fitDataRequired = false;
    for ( int i = 1; i < argc; ++i )
    {
        if( EmbeddedPython::isPythonScript( argv[i] ) )
        {
            EmbeddedPython::init();
            // Draw twice to show all menus on screen
            {
                draw( true );
                draw( true );
            }
            EmbeddedPython::setupArgv( argc - i, &argv[i] );
            EmbeddedPython::runScript( argv[i] );
            // Draw to update after executing script
            {
                draw( true );
            }
            EmbeddedPython::finalize();
            break;
        }
        if( isKnownMeshFile( argv[i] ) )
        {
            if( load_file( argv[i] ) )
                fitDataRequired = true;
        }
    }
    if (fitDataRequired)
        fitDataViewport();
#endif
}

void Viewer::processMouseEventsQueue_()
{
    while ( !mouseEventQueue.empty() )
    {
        mouseEventQueue.front().callEvent();
        mouseEventQueue.pop();
    }
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
}

bool Viewer::isKnownMeshFile( const std::filesystem::path& mesh_file_name )
{
    std::error_code ec;
    if( !std::filesystem::exists( mesh_file_name, ec ) )
        return false;
    if( !std::filesystem::is_regular_file( mesh_file_name, ec ) )
        return false;

    std::string ext = utf8string( mesh_file_name.extension() );
    for( auto& c : ext )
        c = (char) tolower( c );

    auto filts = MeshLoad::getFilters();
    for( auto& filter : filts )
    {
        if( filter.extension.find(ext) )
            return true;
    }

    return false;
}

bool Viewer::load_file( const std::filesystem::path & path )
{
    if ( loadSignal( path ) )
        return true;

    return false;
}

bool Viewer::save_mesh_to_file( const std::filesystem::path & path )
{
    if ( saveSignal( path ) )
        return true;

    return false;
}

bool Viewer::key_pressed( unsigned int unicode_key, int modifiers )
{
    // repeated signals swap each frame to prevent freezes
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, false );

    eventsCounter_.counter[size_t( EventType::CharPressed )]++;

    return charPressedSignal( unicode_key, modifiers );
}

bool Viewer::key_down( int key, int modifiers )
{
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::KeyDown )]++;

    if ( keyDownSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::key_up( int key, int modifiers )
{
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::KeyUp )]++;

    if ( keyUpSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::key_repeat( int key, int modifiers )
{
    // repeated signals swap each frame to prevent freezes
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, false );

    eventsCounter_.counter[size_t( EventType::KeyRepeat )]++;

    if ( keyRepeatSignal( key, modifiers ) )
        return true;

    return false;
}

bool Viewer::mouse_down( MouseButton button, int modifier )
{
    // if the mouse was released in this frame, then we need to render at least one more frame to get button reaction;
    // if the mouse was pressed and released in this frame, then at least two more frames are necessary because of
    // g_MouseJustPressed in ImGui_ImplGlfw_UpdateMousePosAndButtons
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::MouseDown )]++;

    if ( mouseDownSignal( button, modifier ) )
        return true;

    return true;
}

bool Viewer::mouse_up( MouseButton button, int modifier )
{
    // if the mouse was released in this frame, then we need to render at least one more frame to get button reaction;
    // if the mouse was pressed and released in this frame, then at least two more frames are necessary because of
    // g_MouseJustPressed in ImGui_ImplGlfw_UpdateMousePosAndButtons
    incrementForceRedrawFrames( forceRedrawMinimumIncrement_, swapOnLastPostEventsRedraw );

    eventsCounter_.counter[size_t( EventType::MouseUp )]++;

    if ( mouseUpSignal( button, modifier ) )
        return true;

    return true;
}

bool Viewer::mouse_move( int mouse_x, int mouse_y )
{
    eventsCounter_.counter[size_t( EventType::MouseMove )]++;

    if ( mouseMoveSignal( mouse_x, mouse_y ) )
        return true;

    return false;
}

bool Viewer::mouse_scroll( float delta_y )
{
    eventsCounter_.counter[size_t( EventType::MouseScroll )]++;

    if ( mouseScrollSignal( scrollForce * delta_y ) )
        return true;

    return true;
}

bool Viewer::drag_drop( const std::vector<std::filesystem::path>& paths )
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
    for ( const auto& viewport : viewport_list )
        if ( viewport.getRedrawFlag() )
            return true;

    return getRedrawFlagRecursive( SceneRoot::get(), presentViewportsMask_ );
}

void Viewer::resetRedraw_() const
{
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

    if ( obj->getBackColor().a < 255 || obj->getFrontColor( obj->isSelected() ).a < 255 )
        return VisualObjectRenderType::Transparent;

    return VisualObjectRenderType::Opaque;
}

void Viewer::recursiveDraw_( const Viewport& vp, const Object& obj, const AffineXf3f& parentXf, VisualObjectRenderType renderType, int* numDraws ) const
{
    if ( !obj.isVisible( vp.id ) )
        return;
    auto xfCopy = parentXf * obj.xf();
    auto visObj = obj.asType<VisualObject>();
    if ( visObj && ( renderType == getObjRenderType_( visObj, vp.id ) ) )
    {
        bool alphaNeed = renderType == VisualObjectRenderType::Transparent && alphaSortEnabled_;
        vp.draw( *visObj, xfCopy, false, alphaNeed );
        if ( numDraws )
            ++( *numDraws );
    }
    for ( const auto& child : obj.children() )
        recursiveDraw_( vp, *child, xfCopy, renderType, numDraws );
}

void Viewer::draw( bool force )
{
    if ( !force && !needRedraw_() )
        return;

    if ( !isInDraw_ )
        isInDraw_ = true;
    else
    {
        spdlog::error( "Recursive draw call is not allowed" );
        assert( false );
        // if this happens try to use CommandLoop instead of in draw call
        return;
    }

    frameCounter_.startDraw();

    glPrimitivesCounter_.reset();

    setupScene();
    preDrawSignal();
    drawScene();
    postDrawSignal();

    if ( forceRedrawFrames_ > 0 )
    {
        // everything was rendered, reduce the counter
        --forceRedrawFrames_;
    }
    bool swapped = false;
    if ( window )
    {
        swapped = !swapOnLastForcedFrameOnly_ || forceRedrawFrames_ <= 0;
        if ( swapped )
        {
            glfwSwapBuffers( window );
            swapOnLastForcedFrameOnly_ = false;
        }
    }
    frameCounter_.endDraw( swapped );
    isInDraw_ = false;
}

void Viewer::drawScene() const
{
    if ( alphaSortEnabled_ )
        alphaSorter_->clearTransparencyTextures();

    int numTransparent = 0;
    for ( const auto& viewport : viewport_list )
    {
        viewport.preDraw();
        recursiveDraw_( viewport, SceneRoot::get(), AffineXf3f(), VisualObjectRenderType::Opaque );
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

    for ( const auto& viewport : viewport_list )
        viewport.postDraw();

    resetRedraw_();
}

void Viewer::setupScene() const
{
    for ( const auto& viewport : viewport_list )
    {
        viewport.clear_framebuffers();
        viewport.setupView();
    }
}

void Viewer::resize( int w, int h )
{
    if ( window )
    {
        glfwSetWindowSize( window, w, h );
    }
    post_resize( w, h );
}

void Viewer::post_resize( int w, int h )
{
    if ( w == 0 || h == 0 )
        return;
    if ( w == window_width && h == window_width )
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
                rect.min.x = float( rect.min.x / window_width ) * w;
                rect.min.y = float( rect.min.y / window_height ) * h;
                rect.max.x = rect.min.x + float( width( rect ) / window_width ) * w;
                rect.max.y = rect.min.y + float( height( rect ) / window_height ) * h;
                viewport.setViewportRect( rect );
            }
    }
    postResizeSignal( w, h );
    if ( !windowMaximized ) // resize is called after maximized
        windowSaveSize = { w,h };
    if ( w != 0 )
        window_width = w;
    if ( h != 0 )
        window_height = h;


    if ( alphaSorter_ )
        alphaSorter_->updateTransparencyTexturesSize( window_width, window_height );

    if ( isLaunched_ )
        draw();
}

void Viewer::postSetPosition( int xPos, int yPos )
{
    if ( !windowMaximized )
    {
        if ( yPos == 0 )
        {
            assert( false ); // to catch it once it happens
            yPos = 40; // handle for one rare issue
        }
        windowSavePos = { xPos,yPos };
    }
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
    // it is needed ImGui to correctly capture events after refocusing
    if ( focused && focusRedrawReady_ && !isInDraw_ ) 
        MR::Viewer::instanceRef().draw( true );
}

void Viewer::post_rescale( float x, float y )
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
    globalBasisAxes->setLabels( { {"X",1.1f * Vector3f::plusX()},{"Y",1.1f * Vector3f::plusY()},{"Z",1.1f * Vector3f::plusZ()} } );
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
    Vector<Color, FaceId> colorMap( numF );
    const auto arrowSize = numF / 3;
    for (int i = 0; i < arrowSize; i++)
    {
        colorMap[FaceId( i )] = colorX;
        colorMap[FaceId( i + arrowSize )] = colorY;
        colorMap[FaceId( i + arrowSize * 2 )] = colorZ;
    }
    const float labelPos = size + 0.2f;
    auto addLabel = [&] ( const std::string& str, const Vector3f& pos )
    {
        std::shared_ptr<ObjectLabel> label = std::make_shared<ObjectLabel>();
        label->setFrontColor( Color::white(), false );
        label->setLabel( { str, pos } );
        label->setPivotPoint( Vector2f( 0.5f, 0.5f ) );
        label->setVisualizeProperty( false, VisualizeMaskType::DepthTest, ViewportMask::all() );
        basisAxes->addChild( label );
    };
    addLabel( "X", labelPos * Vector3f::plusX() );
    addLabel( "Y", labelPos * Vector3f::plusY() );
    addLabel( "Z", labelPos * Vector3f::plusZ() );
    
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

bool Viewer::windowShouldClose()
{
    if ( !( window && glfwWindowShouldClose( window ) ) && !stopEventLoop_ )
        return false;

    if ( !interruptWindowClose() )
        return true;

    if ( window )
    {
        glfwSetWindowShouldClose( window, false );
    }
    else
    {
        stopEventLoop_ = false;
    }
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
             ( ( window_height - currentPos.y ) > rect.min.y ) &&
             ( ( window_height - currentPos.y ) < rect.min.y + height( rect ) ) )
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

void Viewer::setMinimumForceRedrawFramesAfterEvents( int minimumIncrement )
{
    forceRedrawMinimumIncrement_ = std::max( 2, minimumIncrement );
}

void Viewer::incrementForceRedrawFrames( int i /*= 1 */, bool swapOnLastOnly /*= false */)
{
    if ( isInDraw_ )
        ++i;
    forceRedrawFrames_ = std::max( i, forceRedrawFrames_ );
    swapOnLastForcedFrameOnly_ = swapOnLastOnly;
}

bool Viewer::isCurrentFrameSwapping() const
{
    return !swapOnLastForcedFrameOnly_ || forceRedrawFrames_ <= 1;
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

Image Viewer::captureScreenShot( const Vector2i& pos /*= Vector2i()*/, const Vector2i& sizeP /*= Vector2i()*/ )
{
    Vector2i size = sizeP;
    if ( !size.x )
        size.x = window_width - pos.x;
    else
        size.x = std::min( window_width - pos.x, size.x );

    if ( !size.y )
        size.y = window_height - pos.y;
    else
        size.y = std::min( window_height - pos.y, size.y );
        
    std::vector<Color> pixels( size.x * size.x );

    setupScene();
    drawScene();

    if ( glInitialized_ )
    {
        GL_EXEC( glReadPixels( pos.x, pos.y, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, ( void* )( pixels.data() ) ) );
    }

    return Image{ pixels, size };
}

void Viewer::captureUIScreenShot( std::function<void( const Image& )> callback,
                                  const Vector2i& pos /*= Vector2i()*/, const Vector2i& sizeP /*= Vector2i() */ )
{
    CommandLoop::appendCommand( [callback, pos, sizeP, this] ()
    {
        Vector2i size = sizeP;
        if ( !size.x )
            size.x = window_width - pos.x;
        else
            size.x = std::min( window_width - pos.x, size.x );

        if ( !size.y )
            size.y = window_height - pos.y;
        else
            size.y = std::min( window_height - pos.y, size.y );

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
    return { screenPoint.x - rect.min.x, screenPoint.y + rect.min.y + height( rect ) - window_height, screenPoint.z };
}

Vector3f Viewer::viewportToScreen( const Vector3f& viewportPoint, ViewportId id ) const
{
    if( (presentViewportsMask_ & id).empty() )
        return { 0.f, 0.f, 0.f };

    const auto& rect = viewport( id ).getViewportRect();
    return { viewportPoint.x + rect.min.x, viewportPoint.y - rect.min.y - height( rect ) + window_height, viewportPoint.z };
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

void Viewer::onSceneSaved( const std::filesystem::path& savePath )
{
    if ( !savePath.empty() )
        recentFilesStore.storeFile( savePath );

    if (!SceneFileFilters.empty() && savePath.extension() == SceneFileFilters.front().extension.substr(1))
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

void Viewer::FrameCounter::startDraw()
{
    startDrawTime_ = std::chrono::time_point_cast< std::chrono::milliseconds >( std::chrono::system_clock::now() ).time_since_epoch().count();
}

void Viewer::FrameCounter::endDraw( bool swapped )
{
    if ( swapped )
        ++swappedFrameCounter;
    ++totalFrameCounter;
    auto nowTP = std::chrono::system_clock::now();
    auto nowSec = std::chrono::time_point_cast<std::chrono::seconds>( nowTP ).time_since_epoch().count();
    drawTimeMilliSec = std::chrono::time_point_cast< std::chrono::milliseconds >( nowTP ).time_since_epoch().count() - startDrawTime_;
    if ( nowSec > startFPSTime_ )
    {
        startFPSTime_ = nowSec;
        fps = swappedFrameCounter - startFrameNum;
        startFrameNum = swappedFrameCounter;
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
