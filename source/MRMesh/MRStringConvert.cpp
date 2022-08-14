#include "MRStringConvert.h"
#include <codecvt>
#include <locale>
#include "MRPch/MRSpdlog.h"
#ifdef _WIN32
#include "windows.h"
#endif

namespace MR
{

std::wstring utf8ToWide( const char* utf8 )
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes( utf8 );
}

#ifdef _WIN32
std::u8string Utf16ToUtf8( const std::wstring_view & utf16 )
{
    std::string u8msg;
    u8msg.resize( 2 * utf16.size() + 1 );
    auto res = WideCharToMultiByte( CP_UTF8, 0, utf16.data(), (int)utf16.size(), u8msg.data(), int( u8msg.size() ), NULL, NULL );
    if ( res == 0 )
    {
        spdlog::error( GetLastError() );
        return {};
    }
    u8msg.resize( res );
    return asU8String( u8msg );
}
#endif

std::u8string systemToUtf8( const char* system )
{
    if ( !system )
        return {};
#ifdef _WIN32
    std::string msg = system;
    std::wstring wmsg;
    wmsg.resize( msg.size() + 1 );
    auto res = MultiByteToWideChar( CP_ACP, MB_PRECOMPOSED, msg.c_str(), -1, wmsg.data(), int( wmsg.size() ) );
    if ( res == 0 )
    {
        spdlog::error( GetLastError() );
        return {};
    }
    return Utf16ToUtf8( wmsg );
#else
    return asU8String( system );
#endif
}

std::string bytesString( size_t size )
{
    if ( size < 1024 )
        return fmt::format( "{} bytes", size );
    if ( size < 1024*1024 )
        return fmt::format( "{:.2f} Kb", size / 1024.f );
    if ( size < 1024*1024*1024 )
        return fmt::format( "{:.2f} Mb", size / float(1024*1024) );
    return fmt::format( "{:.2f} Gb", size / float(1024*1024*1024) );
}

}
