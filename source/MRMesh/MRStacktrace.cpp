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
#ifndef _WIN32
    // Catch the remaining default-terminate POSIX signals so a silent kill
    // writes a "Crash signal: N" line + stacktrace before exit, instead of
    // disappearing without a trace.
    std::signal( SIGHUP,  crashSignalHandler );
    std::signal( SIGQUIT, crashSignalHandler );
    std::signal( SIGBUS,  crashSignalHandler );
    std::signal( SIGSYS,  crashSignalHandler );
    std::signal( SIGUSR1, crashSignalHandler );
    std::signal( SIGUSR2, crashSignalHandler );
    // cpp-httplib relies on SIGPIPE being ignored process-wide so socket
    // writes to a disconnected peer return EPIPE instead of terminating the
    // process. Set it here explicitly so the disposition can't be lost to
    // load-order surprises with any third-party code that touches signal().
    std::signal( SIGPIPE, SIG_IGN );
#endif
}

} // namespace MR

#endif
