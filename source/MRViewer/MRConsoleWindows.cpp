#include "MRConsoleWindows.h"

#ifdef _WIN32
#include <Windows.h>
#include <shellapi.h>
#include <MRMesh/MRStringConvert.h>
#include <iostream>

namespace MR
{

std::vector<std::string> ConvertArgv()
{
    int argc = 0;

    LPWSTR* argvW = CommandLineToArgvW( GetCommandLineW(), &argc );
    if ( !argvW )
    {
        assert( false );
        return {};
    }

    std::vector<std::string> arguments;
    arguments.reserve( argc );
    for ( int i = 0; i < argc; ++i )
        arguments.push_back( MR::Utf16ToUtf8( argvW[i] ) );
    return arguments;
}

ConsoleRunner::ConsoleRunner( bool runConsole ):
    consoleStarted_{ runConsole }
{
    if ( !consoleStarted_ )
        return;
    // alloc
    if ( !AllocConsole() )
    {
        consoleStarted_ = false;
        assert( false );
        return;
    }

    // adjust
    constexpr int16_t cConsoleMinLength = 1024;
    // Set the screen buffer to be big enough to scroll some text
    CONSOLE_SCREEN_BUFFER_INFO conInfo;
    GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ), &conInfo );
    if ( conInfo.dwSize.Y < cConsoleMinLength )
        conInfo.dwSize.Y = cConsoleMinLength;
    SetConsoleScreenBufferSize( GetStdHandle( STD_OUTPUT_HANDLE ), conInfo.dwSize );

    // redirect
    bool result = true;
    FILE* fp;

    // Redirect STDIN if the console has an input handle
    if ( GetStdHandle( STD_INPUT_HANDLE ) != INVALID_HANDLE_VALUE )
        if ( freopen_s( &fp, "CONIN$", "r", stdin ) != 0 )
            result = false;
        else
            setvbuf( stdin, NULL, _IONBF, 0 );

    // Redirect STDOUT if the console has an output handle
    if ( GetStdHandle( STD_OUTPUT_HANDLE ) != INVALID_HANDLE_VALUE )
        if ( freopen_s( &fp, "CONOUT$", "w", stdout ) != 0 )
            result = false;
        else
            setvbuf( stdout, NULL, _IONBF, 0 );

    // Redirect STDERR if the console has an error handle
    if ( GetStdHandle( STD_ERROR_HANDLE ) != INVALID_HANDLE_VALUE )
        if ( freopen_s( &fp, "CONOUT$", "w", stderr ) != 0 )
            result = false;
        else
            setvbuf( stderr, NULL, _IONBF, 0 );

    // Make C++ standard streams point to console as well.
    std::ios::sync_with_stdio( true );

    // Clear the error state for each of the C++ standard streams.
    std::wcout.clear();
    std::cout.clear();
    std::wcerr.clear();
    std::cerr.clear();
    std::wcin.clear();
    std::cin.clear();

    SetConsoleOutputCP( CP_UTF8 );

    if ( !result )
    {
        assert( false );
        consoleStarted_ = false;
    }
}

ConsoleRunner::~ConsoleRunner()
{
    if ( !consoleStarted_ )
        return;

    bool result = true;
    FILE* fp;

    // Just to be safe, redirect standard IO to NUL before releasing.

    // Redirect STDIN to NUL
    if ( freopen_s( &fp, "NUL:", "r", stdin ) != 0 )
        result = false;
    else
        setvbuf( stdin, NULL, _IONBF, 0 );

    // Redirect STDOUT to NUL
    if ( freopen_s( &fp, "NUL:", "w", stdout ) != 0 )
        result = false;
    else
        setvbuf( stdout, NULL, _IONBF, 0 );

    // Redirect STDERR to NUL
    if ( freopen_s( &fp, "NUL:", "w", stderr ) != 0 )
        result = false;
    else
        setvbuf( stderr, NULL, _IONBF, 0 );

    // Detach from console
    if ( !FreeConsole() )
        result = false;

    assert( result );
}

} //namespace MR
#endif //_WIN32
