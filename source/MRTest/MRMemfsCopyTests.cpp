#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRPch/MRSpdlog.h"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#if defined( __EMSCRIPTEN__ ) && defined( __EMSCRIPTEN_PTHREADS__ )
#include <emscripten/emscripten.h>
#include <emscripten/threading.h>
#endif

namespace MR
{

namespace
{

#ifdef __EMSCRIPTEN__
constexpr int cSeconds = 480;
#else
constexpr int cSeconds = 2;
#endif

constexpr int cFileKiB = 100;

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )

constexpr int cStallSeconds = 60;

/// keeps this thread awake without parking it on a futex; on the emscripten main thread it
/// also drains the proxying queue, which is what the app does between frames
void stayAwake( int ms )
{
#if defined( __EMSCRIPTEN__ ) && defined( __EMSCRIPTEN_PTHREADS__ )
    emscripten_thread_sleep( ms );
#else
    std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
#endif
}

/// a wedged thread cannot be joined, so the process has to leave without it
[[noreturn]] void leaveNow( int code )
{
#ifdef __EMSCRIPTEN__
    emscripten_force_exit( code );
#endif
    std::_Exit( code );
}

#endif

} // namespace

// Reduced from the application-level reproducer, where a worker doing nothing but
// std::filesystem::copy wedged 3 of 8 CI shards within 8 minutes: the copy stops returning and
// the main thread stops answering with it, while the compositor keeps painting. Everything else
// - zip, scene loading, the converter - turned out to be irrelevant, so this is the whole
// mechanism in one call.
TEST( MRMesh, MemfsCopyStress )
{
    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const std::filesystem::path dir = folder;

    const auto src = dir / "src.bin";
    {
        std::ofstream ofs( src, std::ios::binary );
        ASSERT_TRUE( bool( ofs ) );
        const std::string chunk( 1024, 'x' );
        for ( int i = 0; i < cFileKiB; ++i )
            ofs << chunk;
    }

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    std::atomic<long long> copies{ 0 };
    std::atomic<bool> finished{ false };

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( cSeconds );
    std::thread worker( [&]
    {
        std::error_code ec;
        const auto dst = dir / "dst.bin";
        while ( std::chrono::steady_clock::now() < deadline )
        {
            std::filesystem::remove( dst, ec );
            std::filesystem::copy( src, dst, ec );
            copies.fetch_add( 1, std::memory_order_relaxed );
        }
        finished.store( true, std::memory_order_release );
    } );

    long long seen = 0;
    int ticks = 0;
    auto lastProgress = std::chrono::steady_clock::now();
    while ( !finished.load( std::memory_order_acquire ) )
    {
        stayAwake( 250 );

        const long long now = copies.load( std::memory_order_relaxed );
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
                spdlog::error( "STALLED: no copy finished for {} s after {} copies", idle, seen );
                leaveNow( 3 );
            }
        }

        if ( ++ticks % 20 == 0 )
            spdlog::info( "{} copies", seen );
    }

    worker.join();
    spdlog::info( "done: {} copies in {} s", seen, cSeconds );
    EXPECT_GT( seen, 0 );
#else
    std::error_code ec;
    const auto dst = dir / "dst.bin";
    long long copies = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( cSeconds );
    while ( std::chrono::steady_clock::now() < deadline )
    {
        std::filesystem::remove( dst, ec );
        std::filesystem::copy( src, dst, ec );
        ++copies;
    }
    spdlog::info( "done: {} single-threaded copies", copies );
    EXPECT_GT( copies, 0 );
#endif
}

} // namespace MR
