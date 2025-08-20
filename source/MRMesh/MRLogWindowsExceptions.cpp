#if defined _WIN32 && defined NDEBUG

#include "MRStringConvert.h"
#include "MRSystem.h"
#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWinapi.h"

namespace MR
{

namespace
{

constexpr auto EXCEPTION_CXX   = 0xE06D7363L; // c++ throw ...
constexpr auto MS_VC_EXCEPTION = 0x406D1388L; // thread renaming
constexpr auto RPC_UNAVAILABLE = 0x000006BAL; // thrown in file dialog

// we limit the number of logged stacktraces since typically only the first exception is of any interest,
// and the following ones are either repetitions or consequences of the first exception which only make the log huge
int numMoreStacktraces = 5;

LONG WINAPI logWindowsException( LPEXCEPTION_POINTERS pExInfo )
{
    thread_local bool logging = false;
    if ( logging )
        return EXCEPTION_CONTINUE_SEARCH; //avoid recursive logging

    if ( !pExInfo )
        return EXCEPTION_CONTINUE_SEARCH;
    PEXCEPTION_RECORD pExceptionRecord = pExInfo->ExceptionRecord;

    if ( pExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT ||
         pExceptionRecord->ExceptionCode == EXCEPTION_SINGLE_STEP ||
         pExceptionRecord->ExceptionCode == RPC_UNAVAILABLE ||
         pExceptionRecord->ExceptionCode == EXCEPTION_CXX ||
         pExceptionRecord->ExceptionCode == MS_VC_EXCEPTION )
        return EXCEPTION_CONTINUE_SEARCH; //normal situation, handled otherwise

    logging = true;
    if ( pExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && pExceptionRecord->NumberParameters >= 2 )
    {
        // https://stackoverflow.com/a/22850748/7325599
        auto type = []( ULONG64 v )
        {
            if ( v == 0 )
                return "reading";
            if ( v == 1 )
                return "writing";
            if ( v == 8 )
                return "DEP";
            return "???";
        };
        spdlog::critical( "Access violation: {} at {:#010x}",
            type( pExceptionRecord->ExceptionInformation[0] ),
            pExceptionRecord->ExceptionInformation[1] );
    }
    else if ( pExceptionRecord->ExceptionCode == DBG_PRINTEXCEPTION_C && pExceptionRecord->NumberParameters >= 2 )
    {
        // https://stackoverflow.com/a/41480827/7325599
        auto len = pExceptionRecord->ExceptionInformation[0];
        if ( len )
            --len;
        const auto * p = (PCSTR)pExceptionRecord->ExceptionInformation[1];
        spdlog::info( "Narrow debug information: {}", std::string_view( p, len ) );
    }
    else if ( pExceptionRecord->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C && pExceptionRecord->NumberParameters >= 2 )
    {
        // https://stackoverflow.com/a/41480827/7325599
        auto len = pExceptionRecord->ExceptionInformation[0];
        if ( len )
            --len;
        const auto * p = (PCWSTR)pExceptionRecord->ExceptionInformation[1];
        spdlog::info( "Wide debug information: {}", Utf16ToUtf8( std::wstring_view( p, len ) ) );
    }
    else
        spdlog::warn( "Windows exception {:#010x}", pExceptionRecord->ExceptionCode );

    if ( numMoreStacktraces > 0 )
    {
        --numMoreStacktraces;
        spdlog::info( "Windows exception stacktrace:\n{}", getCurrentStacktrace() );
        printCurrentTimerBranch();
    }
    logging = false;
    return EXCEPTION_CONTINUE_SEARCH;
}

class WindowsExceptionsLogger
{
public:
    WindowsExceptionsLogger();
    ~WindowsExceptionsLogger();

private:
    _purecall_handler oldPurecallHandler_ = nullptr;
    PVOID oldVectoredExceptionHandler_ = nullptr;
} windowsExceptionsLogger_;

WindowsExceptionsLogger::WindowsExceptionsLogger()
{
    spdlog::debug( "Start logging Windows exceptions" );
    oldPurecallHandler_ = _set_purecall_handler( []()
    {
        spdlog::critical( "Pure virtual function call" );
        spdlog::info( "Pure virtual function call stacktrace:\n{}", getCurrentStacktrace() );
        printCurrentTimerBranch();
        std::exit( 0 );
    } );
    // The system does not display the critical-error-handler message box
    // The system does not display the Windows Error Reporting dialog
    SetErrorMode( SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX );
    oldVectoredExceptionHandler_ = AddVectoredExceptionHandler( 0, logWindowsException );
}

WindowsExceptionsLogger::~WindowsExceptionsLogger()
{
    spdlog::debug( "Stop logging Windows exceptions" );
    _set_purecall_handler( oldPurecallHandler_ );
    RemoveVectoredExceptionHandler( oldVectoredExceptionHandler_ );
}

} //anonymous namespace

} //namespace MR

#endif // defined _WIN32 && defined NDEBUG
