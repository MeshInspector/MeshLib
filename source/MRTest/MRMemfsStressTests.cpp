#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRPch/MRSpdlog.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined( __EMSCRIPTEN__ ) && defined( __EMSCRIPTEN_PTHREADS__ )
#include <emscripten/threading.h>
#endif

namespace MR
{

namespace
{

#ifdef __EMSCRIPTEN__
constexpr int cSeconds = 480;
#else
// desktop CI runs this file too, and the target is a wasm-only suspicion there
constexpr int cSeconds = 2;
#endif

/// entry sizes taken from the archives the app has stalled on
constexpr size_t cSizes[] = { 195, 78603, 245796, 1021928 };

/// one round of exactly what the zip paths ask of the filesystem: make sure the parent
/// exists, write a file, read it back, drop it
bool fileRound( const std::filesystem::path& dir, int tid, long long i, std::vector<char>& buf )
{
    std::error_code ec;
    const auto sub = dir / ( "t" + std::to_string( tid ) );
    if ( !std::filesystem::exists( sub, ec ) )
        std::filesystem::create_directories( sub, ec );

    const size_t size = cSizes[i % 4];
    const auto path = sub / ( std::to_string( i % 4 ) + ".bin" );
    buf.assign( size, char( i ) );

    {
        std::ofstream ofs( path, std::ios::binary );
        if ( !ofs )
            return false;
        if ( !ofs.write( buf.data(), (std::streamsize)buf.size() ) )
            return false;
        ofs.close();
    }
    {
        std::ifstream ifs( path, std::ios::binary );
        if ( !ifs )
            return false;
        ifs.read( buf.data(), (std::streamsize)buf.size() );
        if ( (size_t)ifs.gcount() != size )
            return false;
    }
    std::filesystem::remove( path, ec );
    return true;
}

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )

constexpr int cStallSeconds = 60;

/// mimics spdlog's rotating_file_sink_mt with flush_on( info ): one file, held open, one
/// write plus flush per message, on a thread other than the one writing the zip entries.
/// The app has exactly this pair and the singlethreaded build has neither.
void appendForever( const std::filesystem::path& path, const std::atomic<bool>& stop,
    std::atomic<long long>& lines )
{
    std::ofstream ofs( path, std::ios::binary | std::ios::app );
    if ( !ofs )
        return;
    while ( !stop.load( std::memory_order_acquire ) )
    {
        ofs << "[info] a line of about the length the app writes, with a path in it" << char( 10 );
        ofs.flush();
        lines.fetch_add( 1, std::memory_order_relaxed );
    }
}

/// PTHREAD_POOL_SIZE is navigator.hardwareConcurrency, and a thread past that one needs a
/// Worker spawned on demand, which needs the main thread to reach the browser event loop --
/// which this test's watchdog never does. Asking for more than the pool holds therefore
/// deadlocks the test itself and says nothing about the filesystem. The appender takes one
/// slot, so the file workers get the rest.
int poolSizedThreadCount()
{
    const unsigned hc = std::thread::hardware_concurrency();
    return std::max( 1, int( hc ? hc : 4 ) - 1 );
}

/// keeps this thread awake without parking it on a futex; on the emscripten main thread
/// it also drains the proxying queue, which is what the app's UI thread does between frames
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

// Every stall seen in the field has been a filesystem call on a worker thread: twice inside
// compressZip, whose ParallelFor puts several threads in the filesystem at once, and four
// times inside decompressZip, once pinned to the ofstream constructor and once to ofs.write.
// The singlethreaded build has never stalled. So this hammers MEMFS from more threads than
// the runner has cores, with no zip and no scene code in the way: if it wedges, the
// reproducer is small enough to hand upstream.
TEST( MRMesh, MemfsConcurrentStress )
{
    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const std::filesystem::path dir = folder;

#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    std::atomic<long long> done{ 0 };
    std::atomic<int> finished{ 0 };
    std::atomic<bool> broken{ false };
    const int cThreads = poolSizedThreadCount();
    spdlog::info( "{} threads, {} s", cThreads, cSeconds );
    std::vector<std::atomic<long long>> perThread( cThreads );

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( cSeconds );
    std::vector<std::thread> threads;
    threads.reserve( size_t( cThreads ) );
    for ( int tid = 0; tid < cThreads; ++tid )
    {
        threads.emplace_back( [&, tid]
        {
            std::vector<char> buf;
            for ( long long i = 0; std::chrono::steady_clock::now() < deadline && !broken.load( std::memory_order_acquire ); ++i )
            {
                if ( !fileRound( dir, tid, i, buf ) )
                {
                    broken.store( true, std::memory_order_release );
                    break;
                }
                perThread[tid].fetch_add( 1, std::memory_order_relaxed );
                done.fetch_add( 1, std::memory_order_relaxed );
            }
            finished.fetch_add( 1, std::memory_order_release );
        } );
    }

    std::atomic<bool> stopAppender{ false };
    std::atomic<long long> logLines{ 0 };
    std::thread appender( appendForever, dir / "shared.log", std::cref( stopAppender ), std::ref( logLines ) );

    long long seen = 0;
    int ticks = 0;
    auto lastProgress = std::chrono::steady_clock::now();
    while ( finished.load( std::memory_order_acquire ) < cThreads )
    {
        stayAwake( 250 );

        const long long now = done.load( std::memory_order_relaxed );
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
                spdlog::error( "STALLED: no filesystem progress for {} s after {} rounds, {} log lines",
                    idle, seen, logLines.load( std::memory_order_relaxed ) );
                for ( int tid = 0; tid < cThreads; ++tid )
                    spdlog::error( "  thread {}: {} rounds, finished {}", tid,
                        perThread[tid].load( std::memory_order_relaxed ),
                        finished.load( std::memory_order_relaxed ) );
                leaveNow( 3 );
            }
        }

        // once every five seconds, so a stalled log shows this thread alive next to silent workers
        if ( ++ticks % 20 == 0 )
            spdlog::info( "{} rounds across {} threads", seen, cThreads );
    }

    stopAppender.store( true, std::memory_order_release );
    appender.join();
    for ( auto& t : threads )
        t.join();

    ASSERT_FALSE( broken.load( std::memory_order_acquire ) ) << "a filesystem call failed";
    spdlog::info( "done: {} rounds in {} s across {} threads, {} shared-log lines",
        seen, cSeconds, cThreads, logLines.load( std::memory_order_relaxed ) );
    EXPECT_GT( seen, 0 );
#else
    std::vector<char> buf;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( cSeconds );
    long long i = 0;
    for ( ; std::chrono::steady_clock::now() < deadline; ++i )
        ASSERT_TRUE( fileRound( dir, 0, i, buf ) ) << "round " << i;
    spdlog::info( "done: {} single-threaded rounds", i );
    EXPECT_GT( i, 0 );
#endif
}

} // namespace MR
