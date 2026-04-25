#include "MRMCPGatewaySpawn.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <iostream>

namespace MR::McpGateway
{

#ifdef _WIN32

namespace
{

std::wstring utf8ToWide( const std::string& s )
{
    if ( s.empty() )
        return {};
    int n = MultiByteToWideChar( CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0 );
    std::wstring out( (size_t)n, L'\0' );
    MultiByteToWideChar( CP_UTF8, 0, s.data(), (int)s.size(), out.data(), n );
    return out;
}

// Append @p arg to @p cmdLine, quoted/escaped per CommandLineToArgvW rules.
void appendQuotedArg( std::wstring& cmdLine, const std::wstring& arg )
{
    if ( !arg.empty() && arg.find_first_of( L" \t\n\v\"" ) == std::wstring::npos )
    {
        cmdLine.append( arg );
        return;
    }
    cmdLine.push_back( L'"' );
    for ( auto it = arg.begin();; )
    {
        unsigned numBackslashes = 0;
        while ( it != arg.end() && *it == L'\\' )
        {
            ++it;
            ++numBackslashes;
        }
        if ( it == arg.end() )
        {
            cmdLine.append( numBackslashes * 2, L'\\' );
            break;
        }
        if ( *it == L'"' )
        {
            cmdLine.append( numBackslashes * 2 + 1, L'\\' );
            cmdLine.push_back( *it );
        }
        else
        {
            cmdLine.append( numBackslashes, L'\\' );
            cmdLine.push_back( *it );
        }
        ++it;
    }
    cmdLine.push_back( L'"' );
}

} // anonymous namespace

bool spawnDetached( const std::filesystem::path& exe, const std::vector<std::string>& args )
{
    std::wstring cmdLine;
    appendQuotedArg( cmdLine, exe.wstring() );
    for ( const auto& a : args )
    {
        cmdLine.push_back( L' ' );
        appendQuotedArg( cmdLine, utf8ToWide( a ) );
    }

    STARTUPINFOW si{};
    si.cb = sizeof( si );
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(
        exe.c_str(),
        cmdLine.data(),
        nullptr, nullptr,
        FALSE,
        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        nullptr, nullptr,
        &si, &pi );
    if ( !ok )
    {
        std::cerr << "MRMCPGateway: CreateProcess failed, error " << GetLastError() << "\n";
        return false;
    }
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );
    return true;
}

#else // POSIX

bool spawnDetached( const std::filesystem::path& exe, const std::vector<std::string>& args )
{
    pid_t first = fork();
    if ( first < 0 )
        return false;
    if ( first == 0 )
    {
        // First child: detach into a new session, then fork again so the
        // grandchild has no parent in our process tree (no zombies).
        if ( setsid() < 0 )
            _exit( 127 );
        pid_t second = fork();
        if ( second < 0 )
            _exit( 127 );
        if ( second > 0 )
            _exit( 0 );

        // Grandchild: exec the target.
        std::string exeStr = exe.string();
        std::vector<std::string> argsCopy = args;
        std::vector<char*> argv;
        argv.reserve( argsCopy.size() + 2 );
        argv.push_back( exeStr.data() );
        for ( auto& a : argsCopy )
            argv.push_back( a.data() );
        argv.push_back( nullptr );
        execvp( argv[0], argv.data() );
        _exit( 127 );
    }
    // Parent: reap the first child (which exits immediately after the second fork).
    int status = 0;
    waitpid( first, &status, 0 );
    return WIFEXITED( status ) && WEXITSTATUS( status ) == 0;
}

#endif

} // namespace MR::McpGateway
