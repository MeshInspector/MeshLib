#include "MRMesh/MRCube.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#if defined( __EMSCRIPTEN__ ) && defined( __EMSCRIPTEN_PTHREADS__ )
#include <emscripten/threading.h>
#endif

namespace MR
{

namespace
{

constexpr int cIterations = 2000;

void addMesh( Object& root, std::string name, Mesh mesh )
{
    auto obj = std::make_shared<ObjectMesh>();
    obj->setName( std::move( name ) );
    obj->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    root.addChild( std::move( obj ) );
}

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )

constexpr int cStallSeconds = 120;

// Keeps the calling thread awake without letting it fall asleep on a futex: on the
// emscripten main thread this also drains the proxying queue, which is what the app's
// UI thread does between frames.
void stayAwake( int ms )
{
#if defined( __EMSCRIPTEN__ ) && defined( __EMSCRIPTEN_PTHREADS__ )
    emscripten_thread_sleep( ms );
#else
    std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
#endif
}

// The wedged thread cannot be joined, so the process has to leave without it.
[[noreturn]] void leaveNow( int code )
{
#ifdef __EMSCRIPTEN__
    emscripten_force_exit( code );
#endif
    std::_Exit( code );
}

#endif

} // namespace

// Reloads one and the same scene archive many times, hunting a rare stall seen twice in
// a wasm64 browser build: loadObjectFromFile never returned while the page itself stayed
// responsive. The load therefore runs on a worker thread while this thread stays awake and
// watches it, mirroring the app, where the loader is a pthread and the UI thread keeps
// pumping. A previous round doing the same loads on the main thread found nothing in 1600
// of them, which is why the thread the load runs on is now part of the experiment.
TEST( MRMesh, SceneLoadStress )
{
    Object root;
    root.setName( "Root" );
    addMesh( root, "big", makeUVSphere( 1.0f, 300, 300 ) );
    addMesh( root, "small", makeUVSphere( 1.0f, 16, 16 ) );
    addMesh( root, "tiny", makeCube() );

    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const auto mruPath = folder / "stress.mru";
    const auto saved = serializeObjectTree( root, mruPath );
    ASSERT_TRUE( saved.has_value() ) << saved.error();

    std::error_code ec;
    spdlog::info( "stress.mru: {} bytes", std::filesystem::file_size( mruPath, ec ) );

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    std::atomic<int> done{ 0 };
    std::atomic<bool> broken{ false };
    std::string error;

    std::thread loader( [&]
    {
        for ( int i = 0; i < cIterations; ++i )
        {
            Timer t( "t" );
            auto loaded = loadObjectFromFile( mruPath );
            if ( !loaded )
                error = loaded.error();
            else if ( loaded->objs.empty() )
                error = "no objects loaded";
            if ( !error.empty() )
            {
                broken.store( true, std::memory_order_release );
                return;
            }
            spdlog::info( "iteration {}/{}: {} sec", i + 1, cIterations, t.secondsPassed() );
            done.store( i + 1, std::memory_order_release );
        }
    } );

    int seen = 0;
    int ticks = 0;
    auto lastProgress = std::chrono::steady_clock::now();
    while ( done.load( std::memory_order_acquire ) < cIterations && !broken.load( std::memory_order_acquire ) )
    {
        stayAwake( 250 );

        const int now = done.load( std::memory_order_acquire );
        if ( now != seen )
        {
            seen = now;
            lastProgress = std::chrono::steady_clock::now();
        }
        else
        {
            const auto idle = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - lastProgress ).count();
            if ( idle >= cStallSeconds )
            {
                spdlog::error( "STALLED: the loader made no progress for {} s after {} loads", idle, seen );
                spdlog::error( "this thread is still running, so the stall is in the loader thread alone" );
                leaveNow( 3 );
            }
        }

        // once a second, so a stalled log still shows this thread alive next to a silent loader
        if ( ++ticks % 4 == 0 )
            spdlog::info( "watching: {} loads done", seen );
    }

    loader.join();
    ASSERT_FALSE( broken.load( std::memory_order_acquire ) ) << error;
    EXPECT_EQ( done.load( std::memory_order_acquire ), cIterations );
#else
    for ( int i = 0; i < cIterations; ++i )
    {
        Timer t( "t" );
        auto loaded = loadObjectFromFile( mruPath );
        ASSERT_TRUE( loaded.has_value() ) << "iteration " << i << ": " << loaded.error();
        ASSERT_FALSE( loaded->objs.empty() ) << "iteration " << i;
        spdlog::info( "iteration {}/{}: {} sec", i + 1, cIterations, t.secondsPassed() );
    }
#endif
}

} // namespace MR
