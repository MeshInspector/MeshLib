#include "MRStringConvert.h"
#include <codecvt>
#include <locale>
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRSuppressWarning.h"

#include "MRPch/MRWinapi.h"

namespace MR
{

MR_SUPPRESS_WARNING_PUSH
MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )

std::wstring utf8ToWide( const char* utf8 )
{
    // FIXME: std::wstring_convert will be removed in C++26
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes( utf8 );
}

std::string wideToUtf8( const wchar_t * wide )
{
    if ( !wide )
        return {};
    // FIXME: std::wstring_convert will be removed in C++26
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> conv;
    return conv.to_bytes( wide );
}

MR_SUPPRESS_WARNING_POP

#ifdef _WIN32
std::string Utf16ToUtf8( const std::wstring_view & utf16 )
{
    auto res = WideCharToMultiByte( CP_UTF8, 0, utf16.data(), ( int )utf16.size(), 0, 0, NULL, NULL );
    std::string u8msg( res, '\0' );
    res = WideCharToMultiByte( CP_UTF8, 0, utf16.data(), (int)utf16.size(), u8msg.data(), int( u8msg.size() ), NULL, NULL );
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
#ifdef _WIN32
    if ( msg.empty() )
        return msg;
    auto rsize = MultiByteToWideChar( CP_ACP, MB_PRECOMPOSED, msg.data(), int( msg.size() ), nullptr, 0 );
    std::wstring wmsg;
    wmsg.resize( size_t( rsize ) );
    rsize = MultiByteToWideChar( CP_ACP, MB_PRECOMPOSED, msg.data(), int( msg.size() ), wmsg.data(), int( wmsg.size() ) );
    if ( rsize == 0 )
    {
        spdlog::error( GetLastError() );
        return {};
    }
    wmsg.resize( rsize );
    return Utf16ToUtf8( wmsg );
#else
    return msg;
#endif
}

std::string utf8ToSystem( const std::string & utf8 )
{
#ifdef _WIN32
    auto utf16 = utf8ToWide( utf8.c_str() );
    auto rsize = WideCharToMultiByte( CP_ACP, 0, utf16.data(), ( int )utf16.size(), NULL, 0, NULL, NULL );
    std::string res( size_t ( rsize ), '\0' );
    BOOL usedDefaultChar = FALSE;
    rsize = WideCharToMultiByte( CP_ACP, 0, utf16.data(), (int)utf16.size(), res.data(), int( res.size() ), NULL, &usedDefaultChar );
    if ( usedDefaultChar || rsize == 0 )
    {
        spdlog::error( GetLastError() );
        return {};
    }
    res.resize( rsize );
    return res;
#else
    return utf8;
#endif
}

std::string utf8substr( const char * s, size_t pos, size_t count )
{
    if ( !s )
    {
        assert( false );
        return {};
    }
    return wideToUtf8( utf8ToWide( s ).substr( pos, count ).c_str() );
}

// based on https://github.com/skeeto/branchless-utf8 and ImTextCharFromUtf8 from ImGui
std::pair<char32_t, size_t> utf8ToCodepoint( const char* s, size_t size )
{
    const unsigned char buf[4] = {
        size > 0 ? (unsigned char)s[0] : (unsigned char)0,
        size > 1 ? (unsigned char)s[1] : (unsigned char)0,
        size > 2 ? (unsigned char)s[2] : (unsigned char)0,
        size > 3 ? (unsigned char)s[3] : (unsigned char)0,
    };

    constexpr int lengths[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 4, 0
    };
    const auto len = lengths[buf[0] >> 3];
    const auto read = std::min( size_t( len + !len ), size ); // at least 1 byte

    constexpr int masks[]  = {0x00, 0x7f, 0x1f, 0x0f, 0x07};
    constexpr int shiftc[] = {0, 18, 12, 6, 0};
    char32_t ch;
    ch  = (char32_t)(buf[0] & masks[len]) << 18;
    ch |= (char32_t)(buf[1] & 0x3f) << 12;
    ch |= (char32_t)(buf[2] & 0x3f) <<  6;
    ch |= (char32_t)(buf[3] & 0x3f) <<  0;
    ch >>= shiftc[len];

    constexpr uint32_t mins[] = {4194304, 0, 128, 2048, 65536};
    constexpr int shifte[] = {0, 6, 4, 2, 0};
    int e;
    e  = (ch < mins[len]) << 6; // non-canonical encoding
    e |= ((ch >> 11) == 0x1b) << 7;  // surrogate half?
    e |= (ch > 0x10FFFF) << 8;  // out of range?
    e |= (buf[1] & 0xc0) >> 2;
    e |= (buf[2] & 0xc0) >> 4;
    e |= (buf[3]       ) >> 6;
    e ^= 0x2a; // top two bits of each tail byte correct?
    e >>= shifte[len];
    if ( e )
        ch = U'\xFFFD'; // REPLACEMENT CHARACTER

    return { ch, read };
}

std::u32string utf8ToUtf32( const std::string& str )
{
    std::u32string result;
    for ( size_t cur = 0; cur < str.size(); )
    {
        const auto [ch, read] = utf8ToCodepoint( str.data() + cur, str.size() - cur );
        if ( read == 0 )
            return result;
        result.append( 1, ch );
        cur += read;
    }
    return result;
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

bool hasProhibitedChars( const std::string& line )
{
    for ( const auto& c : line )
        if ( isProhibitedChar( c ) )
            return true;
    return false;
}

std::string replaceProhibitedChars( const std::string& line, char replacement /*= '_' */ )
{
    auto res = line;
    for ( auto& c : res )
        if ( isProhibitedChar( c ) )
            c = replacement;
    return res;
}

std::string commonFilesName( const std::vector<std::filesystem::path> & files )
{
    if ( files.empty() )
        return "Empty";

    if ( std::all_of( files.begin(), files.end(), [&] ( auto&& path ) { std::error_code ec; return is_directory( path, ec ); } ) )
    {
        return files.size() == 1 ? "Directory" : "Directories";
    }

    auto getUpperExt = []( const std::filesystem::path & file )
    {
        auto ext = utf8string( file.extension() );
        for ( auto& c : ext )
            c = ( char )toupper( c );
        return ext;
    };

    auto commonExt = getUpperExt( files[0] );
    if ( files.size() == 1 )
        return commonExt;

    for ( int i = 1; i < files.size(); ++i )
        if ( commonExt != getUpperExt( files[i] ) )
            return "Files";

    commonExt += 's';
    return commonExt;
}

char * formatNoTrailingZeros( char * fmt, double v, int digitsAfterPoint, int precision )
{
    assert( precision > 0 );
    assert( digitsAfterPoint >= 0 && digitsAfterPoint <= 9 );
    double cmp = 1;
    int digitsBeforePoint = 0;
    while ( digitsBeforePoint < precision && std::abs( v ) >= cmp )
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

std::string toLower( std::string str )
{
    for ( auto& ch : str )
        ch = toLower( ch );
    return str;
}

}
