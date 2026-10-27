// Minimal GLFW client: prints what the Wayland compositor tells GLFW about scaling.
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

static void onScale( GLFWwindow* w, float x, float y ) { (void)w; printf( "[%.3f] content scale callback %.2f %.2f\n", glfwGetTime(), x, y ); }
static void onFb( GLFWwindow* w, int a, int b ) { (void)w; printf( "[%.3f] framebuffer size callback %d x %d\n", glfwGetTime(), a, b ); }
static void onErr( int c, const char* d ) { printf( "glfw error %d: %s\n", c, d ); }

static void report( GLFWwindow* w, int i )
{
    int fw, fh, ww, wh;
    float cx, cy;
    glfwGetFramebufferSize( w, &fw, &fh );
    glfwGetWindowSize( w, &ww, &wh );
    glfwGetWindowContentScale( w, &cx, &cy );
    printf( "[%.3f] iteration %d: window %d x %d, framebuffer %d x %d, content scale %.2f\n", glfwGetTime(), i, ww, wh, fw, fh, cx );
}

int main( int argc, char** argv )
{
    const char* mode = argc > 1 ? argv[1] : "default";
    printf( "mode %s\n", mode );
    glfwSetErrorCallback( onErr );
    if ( !glfwInit() )
    {
        puts( "glfwInit failed" );
        return 1;
    }
    printf( "platform %d (GLFW_PLATFORM_WAYLAND = %d)\n", glfwGetPlatform(), GLFW_PLATFORM_WAYLAND );
    GLFWmonitor* m = glfwGetPrimaryMonitor();
    float sx = 0, sy = 0;
    int wx, wy, ww, wh;
    glfwGetMonitorContentScale( m, &sx, &sy );
    glfwGetMonitorWorkarea( m, &wx, &wy, &ww, &wh );
    const GLFWvidmode* vm = glfwGetVideoMode( m );
    printf( "monitor content scale %.2f, workarea %d %d %d x %d, video mode %d x %d\n", sx, wx, wy, ww, wh, vm->width, vm->height );

    if ( strcmp( mode, "twowin" ) == 0 )
    {
        // like the splash window: a second window that lives only during start-up
        glfwWindowHint( GLFW_DECORATED, GLFW_FALSE );
        glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE );
        GLFWwindow* s = glfwCreateWindow( 1, 1, "splash", NULL, NULL );
        glfwMakeContextCurrent( s );
        glfwSetWindowSize( s, 900, 500 );
        for ( int i = 0; i < 50; ++i ) { glfwSwapBuffers( s ); glfwWaitEventsTimeout( 0.02 ); }
        glfwDestroyWindow( s );
        glfwDefaultWindowHints();
    }
    glfwWindowHint( GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE );
    if ( strcmp( mode, "hidden" ) == 0 || strcmp( mode, "twowin" ) == 0 )
        glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );
    if ( strcmp( mode, "transparent" ) == 0 )
        glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE );
    glfwWindowHint( GLFW_SAMPLES, 4 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_FOCUS_ON_SHOW, GLFW_TRUE );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    GLFWwindow* w = glfwCreateWindow( 640, 400, "probe", NULL, NULL );
    if ( !w )
    {
        puts( "glfwCreateWindow failed" );
        return 1;
    }
    glfwSetWindowContentScaleCallback( w, onScale );
    glfwSetFramebufferSizeCallback( w, onFb );
    glfwMakeContextCurrent( w );
    report( w, 0 );
    if ( strcmp( mode, "hidden" ) == 0 || strcmp( mode, "twowin" ) == 0 )
    {
        for ( int i = 0; i < 20; ++i ) { glfwPollEvents(); usleep( 10000 ); }
        printf( "[%.3f] glfwShowWindow\n", glfwGetTime() );
        glfwShowWindow( w );
    }
    for ( int i = 1; i <= 300; ++i )
    {
        glfwSwapBuffers( w );
        glfwPollEvents();
        usleep( 10000 );
        if ( i == 1 || i == 10 || i == 100 || i == 300 )
            report( w, i );
    }
    glfwDestroyWindow( w );
    glfwTerminate();
    return 0;
}
