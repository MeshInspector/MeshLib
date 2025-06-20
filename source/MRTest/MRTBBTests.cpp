#include <MRMesh/MRGTest.h>
#include <MRPch/MRSpdlog.h>
#include <MRPch/MRTBB.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#if __has_include(<fmt/std.h>)
#include <fmt/std.h> // This formats `std::thread::id`.
#endif

namespace MR
{

TEST(MRMesh, TBBTask)
{
    const auto numThreads = tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism );
    spdlog::info( "TBB number of threads is {}", numThreads );
    spdlog::info( "Hardware concurrency is {}", std::thread::hardware_concurrency() );

    const auto mainThreadId = std::this_thread::get_id();
    decltype( std::this_thread::get_id() ) taskThreadId;
    tbb::task_group group;
    std::atomic<bool> taskFinished{ false };
    std::mutex mutex;
    std::condition_variable cvar;
    group.run( [&]
    {
        std::unique_lock lock( mutex );
        taskThreadId = std::this_thread::get_id();
        taskFinished = true;
        cvar.notify_one();
    } );

    if ( numThreads > 1 )
    {
        std::unique_lock lock( mutex );
        cvar.wait( lock, [&taskFinished]() { return taskFinished.load(); } );
    }

    group.wait();
    spdlog::info( "Main in thread {}", mainThreadId );
    spdlog::info( "Task in thread {}", taskThreadId );
    const bool sameThread = mainThreadId == taskThreadId;

    EXPECT_TRUE( ( numThreads == 1 && sameThread ) || ( numThreads > 1 && !sameThread ) );
}

} //namespace MR
