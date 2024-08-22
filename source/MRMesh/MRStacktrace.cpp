#include "MRStacktrace.h"

#ifndef __EMSCRIPTEN__

#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"

#include <csignal>

namespace
{

void crashSignalHandler( int signal )
{
    spdlog::critical( "Crash signal: {}", signal );
    spdlog::info( "Crash stacktrace:\n{}", MR::getCurrentStacktraceInline() );
    MR::printCurrentTimerBranch();
    std::exit( signal );
}

}

namespace MR
{

void printStacktraceOnCrash()
{
    std::signal( SIGTERM, crashSignalHandler );
    std::signal( SIGSEGV, crashSignalHandler );
    std::signal( SIGINT, crashSignalHandler );
    std::signal( SIGILL, crashSignalHandler );
    std::signal( SIGABRT, crashSignalHandler );
    std::signal( SIGFPE, crashSignalHandler );
}

} // namespace MR

#endif
