#pragma once

#include "MRMeshFwd.h"
#include <filesystem>
#include <string>
#include "MRExpected.h"
#include "MRPch/MRBindingMacros.h"

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// converts UTF8-encoded string into UTF16-encoded string
[[nodiscard]] MR_BIND_IGNORE MRMESH_API std::wstring utf8ToWide( const char* utf8 );

/// converts system encoded string to UTF8-encoded string
[[nodiscard]] MRMESH_API std::string systemToUtf8( const std::string & system );

/// converts UTF8-encoded string to system encoded string,
/// returns empty string if such conversion cannot be made
[[nodiscard]] MRMESH_API std::string utf8ToSystem( const std::string & utf8 );

/// converts wide null terminating string to UTF8-encoded string
[[nodiscard]] MR_BIND_IGNORE MRMESH_API std::string wideToUtf8( const wchar_t * wide );

#ifdef _WIN32
/// converts UTF16-encoded string to UTF8-encoded string
[[nodiscard]] MR_BIND_IGNORE MRMESH_API std::string Utf16ToUtf8( const std::wstring_view & utf16 );
#endif

#if defined __cpp_lib_char8_t

[[nodiscard]] MR_BIND_IGNORE inline std::string asString( const std::u8string & s ) { return { s.begin(), s.end() }; }
[[nodiscard]] MR_BIND_IGNORE inline std::u8string asU8String( const std::string & s ) { return { s.begin(), s.end() }; }

#if defined( _LIBCPP_VERSION ) && _LIBCPP_VERSION < 12000
[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const std::string & s ) { return std::filesystem::path( s ); }
[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::path( std::string( s ) ); }
#else
[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const std::string & s ) { return std::filesystem::path( asU8String( s ) ); }
[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::path( asU8String( std::string( s ) ) ); }
#endif

#else // std::u8string is not defined

[[nodiscard]] MR_BIND_IGNORE inline const std::string & asString( const std::string & s ) { return s; }
[[nodiscard]] MR_BIND_IGNORE inline const std::string & asU8String( const std::string & s ) { return s; }

[[nodiscard]] MR_BIND_IGNORE inline std::string asString( std::string && s ) { return std::move( s ); }
[[nodiscard]] MR_BIND_IGNORE inline std::string asU8String( std::string && s ) { return std::move( s ); }

[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const std::string & s ) { return std::filesystem::u8path( s ); }
[[nodiscard]] MR_BIND_IGNORE inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::u8path( s ); }

#endif

/// returns filename as UTF8-encoded string
#if defined( _LIBCPP_VERSION ) && _LIBCPP_VERSION < 12000
[[nodiscard]] inline std::string utf8string( const std::filesystem::path & path )
    { return path.u8string(); }
#else
[[nodiscard]] inline std::string utf8string( const std::filesystem::path & path )
    { return asString( path.u8string() ); }
#endif

/// \}

/// converts given size in string:
/// [0,1024) -> nnn bytes
/// [1024,1024*1024) -> nnn.nn Kb
/// [1024*1024,1024*1024*1024) -> nnn.nn Mb
/// ...
[[nodiscard]] MRMESH_API std::string bytesString( size_t size );

/// returns true if line contains any of OS prohibited chars ('?', '*', '/', '\', '"', '<', '>')
[[nodiscard]] MRMESH_API bool hasProhibitedChars( const std::string& line );

/// replace OS prohibited chars ('?', '*', '/', '\', '"', '<', '>') with `replacement` char
[[nodiscard]] MRMESH_API std::string replaceProhibitedChars( const std::string& line, char replacement = '_' );

/// if (v) contains an error, then appends given file name to that error
template<typename T>
[[nodiscard]] inline Expected<T> addFileNameInError( Expected<T> v, const std::filesystem::path & file )
{
    if ( !v.has_value() )
        v = unexpected( v.error() + ": " + utf8string( file ) );
    return v;
}

/// in case of empty vector, returns "Empty"
/// in case of single input file.ext, returns ".EXT"
/// in case of multiple files with same extension, returns ".EXTs"
/// otherwise returns "Files"
[[nodiscard]] MRMESH_API std::string commonFilesName( const std::vector<std::filesystem::path> & files );

/// returns printf-format for floating-point value in decimal notation with given precision in digits
/// and without trailing zeros after the decimal point
/// \param fmt       preallocated buffer of 5 bytes
/// \param v         value to print
/// \param digitsAfterPoint  maximal number of digits after decimal point
/// \param precision         maximal number of not-zero decimal digits
[[deprecated("Use `valueToString()` from `MRViewer/MRUnits.h` instead!")]]
MRMESH_API MR_BIND_IGNORE char * formatNoTrailingZeros( char * fmt, double v, int digitsAfterPoint, int precision = 6 );

/// returns given value rounded to given number of decimal digits
[[nodiscard]] MRMESH_API double roundToPrecision( double v, int precision );

/// returns given value rounded to given number of decimal digits
[[nodiscard]] inline float roundToPrecision( float v, int precision ) { return (float)roundToPrecision( double(v), precision ); }

// Returns message showed when loading is canceled
[[nodiscard]] inline std::string getCancelMessage( const std::filesystem::path& path )
{
    return "Loading canceled: " + utf8string( path );
}

/// return a copy of the string with all alphabetic ASCII characters replaced with upper-case variants
[[nodiscard]] MRMESH_API std::string toLower( std::string str );

} //namespace MR
