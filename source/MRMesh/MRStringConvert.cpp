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
std::string Utf16ToUtf8( const std::wstring_view & utf16 )
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
    return u8msg;
}
#endif

std::string systemToUtf8( const std::string & msg )
{
    if ( msg.empty() )
        return msg;
#ifdef _WIN32
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
    return msg;
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

char * formatNoTrailingZeros( char * fmt, double v, int digitsAfterPoint, int precision )
{
    assert( precision > 0 );
    assert( digitsAfterPoint >= 0 && digitsAfterPoint <= 9 );
    double cmp = 1;
    int digitsBeforePoint = 0;
    while ( digitsBeforePoint < precision && v >= cmp )
    {
        cmp *= 10;
        ++digitsBeforePoint;
    }
    digitsAfterPoint = std::min( digitsAfterPoint, precision - digitsBeforePoint );

    strcpy( fmt, "%.9f" );
    fmt[2] = char( '0' + digitsAfterPoint );
    if ( digitsAfterPoint <= 0 )
        return fmt;

    char buf[32];
#pragma warning(push)
#pragma warning(disable: 4774) // format string expected in argument 3 is not a string literal
    int n = snprintf( buf, 32, fmt, v );
#pragma warning(pop)

    if ( n >= 0 && std::find( buf, buf + n, '.' ) != buf + n )
    {
        while ( buf[--n] == '0' )
            --digitsAfterPoint;
        assert( digitsAfterPoint >= 0 && digitsAfterPoint <= 9 );
        fmt[2] = char( '0' + digitsAfterPoint );
    }
    return fmt;
}

double roundToPrecision( double v, int precision )
{
    assert( precision >= 1 && precision <= 9 );
    char fmt[] = "%.9g";
    fmt[2] = char( '0' + precision );

    char buf[32];
#pragma warning(push)
#pragma warning(disable: 4774) // format string expected in argument 3 is not a string literal
    int n = snprintf( buf, 32, fmt, v );
#pragma warning(pop)

    return n >= 0 ? std::atof( buf ) : v;
}

}
