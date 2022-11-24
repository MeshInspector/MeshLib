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
    Timer( std::string name ) { start( std::move( name ) ); }
    ~Timer() { finish(); }

    MRMESH_API void restart( std::string name );
    MRMESH_API void start( std::string name );
    MRMESH_API void finish();

    Timer( const Timer & ) = delete;
    Timer & operator =( const Timer & ) = delete;
    Timer( Timer && ) = delete;
    Timer & operator =( Timer && ) = delete;

    std::chrono::duration<double> secondsPassed() const { return std::chrono::high_resolution_clock::now() - start_; }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool started_{ false };
};

/// enables or disables printing of timing tree when application terminates
/// \param minTimeSec omit printing records with time spent less than given value in seconds
MRMESH_API void printTimingTreeAtEnd( bool on, double minTimeSec = 0.1 );

/// prints current timer branch
MRMESH_API void printCurrentTimerBranch();

/// prints the current timing tree, then calls printTimingTreeAtEnd( false );
/// \param minTimeSec omit printing records with time spent less than given value in seconds
MRMESH_API void printTimingTreeAndStop( double minTimeSec = 0.1 );

/// \}

} // namespace MR

#define MR_TIMER MR::Timer _timer( __FUNCTION__ );
#define MR_NAMED_TIMER(name) MR::Timer _named_timer( name );
