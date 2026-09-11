// Standalone reproducer for a wasm stall seen in a large application: a worker thread doing
// nothing but std::filesystem::copy eventually stops returning, and the main thread stops with
// it. No application code, no framework -- the main thread only has to be in a real browser
// event loop, which emscripten_set_main_loop gives it.
#include <emscripten.h>
#include <emscripten/html5_webgl.h>

#include <GLES2/gl2.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace
{

constexpr int cSeconds = 480;
constexpr int cStallSeconds = 60;
constexpr int cFileKiB = 100;
/// the application stalls with a heap around this size
constexpr size_t cBallastMiB = 600;

std::atomic<long long> gCopies{ 0 };
std::atomic<bool> gStop{ false };
int gExitCountdown = -1;
std::chrono::steady_clock::time_point gStart;
std::chrono::steady_clock::time_point gLastProgress;
long long gSeen = 0;

int secondsSince( std::chrono::steady_clock::time_point t )
{
    return int( std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t ).count() );
}

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE gGl = 0;

void frame()
{
    if ( gGl )
    {
        glClearColor( 0.1f, 0.1f, 0.1f, 1.0f );
        glClear( GL_COLOR_BUFFER_BIT );
    }

    const long long now = gCopies.load( std::memory_order_relaxed );
    if ( now != gSeen )
    {
        gSeen = now;
        gLastProgress = std::chrono::steady_clock::now();
    }
    else if ( secondsSince( gLastProgress ) >= cStallSeconds )
    {
        std::printf( "STALLED: no copy finished for %d s after %lld copies", cStallSeconds, gSeen );
        std::putchar( 10 );
        std::fflush( stdout );
        emscripten_force_exit( 3 );
    }

    if ( gExitCountdown > 0 )
    {
        if ( --gExitCountdown == 0 )
            emscripten_force_exit( 0 );
        return;
    }

    if ( secondsSince( gStart ) >= cSeconds )
    {
        std::printf( "done: %lld copies in %d s", gSeen, cSeconds );
        std::putchar( 10 );
        std::fflush( stdout );
        // let both threads leave their loops first: exiting with one still running hangs
        gStop.store( true, std::memory_order_release );
        gExitCountdown = 60;
    }
}

} // namespace

int main()
{
    std::printf( "hardware_concurrency %u", std::thread::hardware_concurrency() );
    std::putchar( 10 );
    std::fflush( stdout );

    // a WebGL context and a heap the size of the application's, the two things the
    // application still has that this program did not
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes( &attrs );
    attrs.majorVersion = 2;
    gGl = emscripten_webgl_create_context( "#canvas", &attrs );
    if ( gGl )
        emscripten_webgl_make_context_current( gGl );
    std::printf( "webgl context %d", int( gGl ) );
    std::putchar( 10 );

    auto ballast = static_cast<char*>( std::malloc( cBallastMiB << 20 ) );
    if ( ballast )
        std::memset( ballast, 1, cBallastMiB << 20 );
    std::printf( "ballast %s", ballast ? "allocated" : "FAILED" );
    std::putchar( 10 );
    std::fflush( stdout );

    std::error_code ec;
    const std::filesystem::path dir = "/tmp/repro";
    std::filesystem::create_directories( dir, ec );

    const auto src = dir / "src.bin";
    {
        std::ofstream ofs( src, std::ios::binary );
        const std::string chunk( 1024, 'x' );
        for ( int i = 0; i < cFileKiB; ++i )
            ofs << chunk;
    }

    std::thread( [dir]
    {
        std::ofstream log( dir / "log.txt", std::ios::binary | std::ios::app );
        while ( log && !gStop.load( std::memory_order_acquire ) )
        {
            log << "[info] a line of about the length the application writes";
            log.put( char( 10 ) );
            log.flush();
        }
    } ).detach();

    std::thread( [src, dir]
    {
        std::error_code workerEc;
        const auto dst = dir / "dst.bin";
        while ( !gStop.load( std::memory_order_acquire ) )
        {
            std::filesystem::remove( dst, workerEc );
            std::filesystem::copy( src, dst, workerEc );
            gCopies.fetch_add( 1, std::memory_order_relaxed );
        }
    } ).detach();

    gStart = gLastProgress = std::chrono::steady_clock::now();

    // 0 fps means requestAnimationFrame, and the final 0 means main() returns instead of
    // blocking -- so the main thread really is back in the browser's event loop between frames
    emscripten_set_main_loop( frame, 0, 0 );
    return 0;
}
