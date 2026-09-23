#include "MRStacktrace.h"

#ifndef __EMSCRIPTEN__

#include "MRSystem.h"
#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"

#include <csignal>

#ifdef _WIN32
#include "MRPch/MRWinapi.h"
#include <filesystem>
#endif

namespace
{

#ifdef _WIN32
std::wstring moduleDirectory( HMODULE module )
{
    wchar_t path[MAX_PATH];
    const auto size = GetModuleFileNameW( module, path, MAX_PATH );
    if ( size == 0 || size == MAX_PATH )
        return {};
    return std::filesystem::path( path ).parent_path().wstring();
}

// MSBuild links with /PDBALTPATH:%_PDB%, so the debug engine behind stacktraces looks for PDB files only
// in the current directory and in _NT_SYMBOL_PATH; add the folders of the executable and of MRMesh.dll to the latter
const bool symbolPathExtended = []
{
    std::wstring symbolPath( 32767, L'\0' );
    symbolPath.resize( GetEnvironmentVariableW( L"_NT_SYMBOL_PATH", symbolPath.data(), DWORD( symbolPath.size() ) ) );
    HMODULE mrmesh = nullptr;
    GetModuleHandleExW( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)&moduleDirectory, &mrmesh );
    for ( const auto& dir : { moduleDirectory( nullptr ), moduleDirectory( mrmesh ) } )
    {
        if ( dir.empty() || symbolPath.find( dir ) != std::wstring::npos )
            continue;
        if ( !symbolPath.empty() )
            symbolPath += L';';
        symbolPath += dir;
    }
    return SetEnvironmentVariableW( L"_NT_SYMBOL_PATH", symbolPath.c_str() ) != 0;
}();
#endif

void crashSignalHandler( int signal )
{
    spdlog::critical( "Crash signal: {}", signal );
    spdlog::info( "Crash stacktrace:\n{}", MR::getCurrentStacktrace() );
    MR::printCurrentTimerBranch();
    std::exit( signal );
}

}

namespace MR
{

void printStacktraceOnCrash()
{
    std::signal( SIGABRT, crashSignalHandler );

#ifndef _WIN32
    //on Windows we use WindowsExceptionsLogger instead of the following signals
    std::signal( SIGTERM, crashSignalHandler );
    std::signal( SIGSEGV, crashSignalHandler );
    std::signal( SIGINT, crashSignalHandler );
    std::signal( SIGILL, crashSignalHandler );
    std::signal( SIGFPE, crashSignalHandler );

    // these signals are not present in Microsoft's implementation
    std::signal( SIGHUP,  crashSignalHandler );
    std::signal( SIGQUIT, crashSignalHandler );
    std::signal( SIGBUS,  crashSignalHandler );
    std::signal( SIGSYS,  crashSignalHandler );
    std::signal( SIGUSR1, crashSignalHandler );
    std::signal( SIGUSR2, crashSignalHandler );

    // cpp-httplib relies on SIGPIPE being ignored process-wide so socket
    // writes to a disconnected peer return EPIPE instead of terminating the process.
    std::signal( SIGPIPE, SIG_IGN );
#endif
}

} // namespace MR

#endif
