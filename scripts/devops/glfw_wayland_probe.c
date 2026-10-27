// Minimal GLFW client: prints what the Wayland compositor tells GLFW about scaling.
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <unistd.h>

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

int main( void )
{
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

    glfwWindowHint( GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE );
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
