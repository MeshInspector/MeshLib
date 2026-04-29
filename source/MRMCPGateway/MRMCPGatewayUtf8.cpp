#include "MRMCPGatewayUtf8.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

namespace MR::McpGateway
{

#ifdef _WIN32

namespace
{

// Wide UTF-16 -> narrow UTF-8. Used by both `pathToUtf8` (path -> wstring source)
// and `getUtf8Argv` (CommandLineToArgvW source).
std::string utf16ToUtf8( const wchar_t* w, int len )
{
    if ( len <= 0 )
        return {};
    int n = WideCharToMultiByte( CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr );
    std::string out( static_cast<size_t>( n ), '\0' );
    WideCharToMultiByte( CP_UTF8, 0, w, len, out.data(), n, nullptr, nullptr );
    return out;
}

} // anonymous namespace

#endif // _WIN32

std::filesystem::path pathFromUtf8( const std::string& s )
{
    // On Windows, `path( std::string )` interprets the bytes as the active locale.
    // Constructing through `std::u8string` keeps the bytes labeled as UTF-8 so
    // libstdc++/MSVC routes them through the UTF-8 codec instead.
    return std::filesystem::path( std::u8string( s.begin(), s.end() ) );
}

std::string pathToUtf8( const std::filesystem::path& p )
{
#ifdef _WIN32
    // `path::string()` on Windows narrows via the active locale (CP_ACP), which
    // corrupts non-ASCII paths. Go through the UTF-16 internal representation and
    // encode as UTF-8 explicitly.
    const auto w = p.wstring();
    return utf16ToUtf8( w.data(), static_cast<int>( w.size() ) );
#else
    // POSIX paths are byte sequences; the convention is UTF-8.
    return p.string();
#endif
}

#ifdef _WIN32
std::vector<std::string> getUtf8Argv()
{
    int wargc = 0;
    wchar_t** wargv = CommandLineToArgvW( GetCommandLineW(), &wargc );
    std::vector<std::string> out;
    if ( !wargv )
        return out;
    out.reserve( wargc );
    for ( int i = 0; i < wargc; ++i )
        out.push_back( utf16ToUtf8( wargv[i], static_cast<int>( wcslen( wargv[i] ) ) ) );
    LocalFree( wargv );
    return out;
}
#endif

} // namespace MR::McpGateway
