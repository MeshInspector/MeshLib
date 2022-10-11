#pragma once

#include "MRMeshFwd.h"
#include <filesystem>
#include <string>
#include <tl/expected.hpp>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// converts UTF8-encoded string into UTF16-encoded string
MRMESH_API std::wstring utf8ToWide( const char* utf8 );
/// converts system encoded string to UTF8-encoded string
MRMESH_API std::u8string systemToUtf8( const char* system );

#ifdef _WIN32
/// converts UTF16-encoded string string to UTF8-encoded string
MRMESH_API std::u8string Utf16ToUtf8( const std::wstring_view & utf16 );
#endif

#if defined __cpp_lib_char8_t

inline const std::string & asString( const std::u8string & s ) { return reinterpret_cast<const std::string &>( s ); }
inline const std::u8string & asU8String( const std::string & s ) { return reinterpret_cast<const std::u8string &>( s ); }

inline std::string asString( std::u8string && s ) { return reinterpret_cast<std::string &&>( s ); }
inline std::u8string asU8String( std::string && s ) { return reinterpret_cast<std::u8string &&>( s ); }

#if defined( _LIBCPP_VERSION ) && _LIBCPP_VERSION < 12000
inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::path( std::string( s ) ); }
#else
inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::path( asU8String( std::string( s ) ) ); }
#endif

#else // std::u8string is not defined

inline const std::string & asString( const std::string & s ) { return s; }
inline const std::string & asU8String( const std::string & s ) { return s; }

inline std::string asString( std::string && s ) { return std::move( s ); }
inline std::string asU8String( std::string && s ) { return std::move( s ); }

inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::u8path( s ); }

#endif

/// returns filename as UTF8-encoded string
#if defined( _LIBCPP_VERSION ) && _LIBCPP_VERSION < 12000
inline std::string utf8string( const std::filesystem::path & path )
    { return path.u8string(); }
#else
inline std::string utf8string( const std::filesystem::path & path )
    { return asString( path.u8string() ); }
#endif

/// \}

/// converts given size in string:
/// [0,1024) -> nnn bytes
/// [1024,1024*1024) -> nnn.nn Kb
/// [1024*1024,1024*1024*1024) -> nnn.nn Mb
/// ...
MRMESH_API std::string bytesString( size_t size );

/// if (v) contains an error, then appends given file name to that error
template<typename T>
inline tl::expected<T, std::string> addFileNameInError( tl::expected<T, std::string> v, const std::filesystem::path & file )
{
    if ( !v.has_value() )
        v = tl::make_unexpected( v.error() + ": " + utf8string( file ) );
    return v;
}

}
