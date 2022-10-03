#pragma once

#include "MRMeshFwd.h"
#include <chrono>
#include <string>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

struct TimeRecord;

class Timer
{
public:
    Timer( const std::string& name ) { start( name ); }
    ~Timer() { finish(); }

    MRMESH_API void restart( const std::string& name );
    MRMESH_API void start( const std::string& name );
    MRMESH_API void finish();

    Timer( const Timer & ) = delete;
    Timer & operator =( const Timer & ) = delete;
    Timer( Timer && ) = delete;
    Timer & operator =( Timer && ) = delete;

    std::chrono::duration<double> secondsPassed() const { return std::chrono::high_resolution_clock::now() - start_; }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/// enables or disables printing of timing tree when application terminates
MRMESH_API void printTimingTreeAtEnd( bool on );

/// prints current timer branch
MRMESH_API void printCurrentTimerBranch();

/// prints the current timing tree, then calls printTimingTreeAtEnd( false );
MRMESH_API void printTimingTreeAndStop();

/// \}

} // namespace MR

#define MR_TIMER MR::Timer _timer( __FUNCTION__ );
#define MR_NAMED_TIMER(name) MR::Timer _named_timer( name );
