#pragma once

#include "MRMeshFwd.h"
#include <filesystem>
#include <string>

namespace MR
{

// converts UTF8-encoded string into UTF16-encoded string
MRMESH_API std::wstring utf8ToWide( const char* utf8 );
// converts system encoded string to UTF8-encoded string
MRMESH_API std::u8string systemToUtf8( const char* system );

#ifdef _WIN32
// converts UTF16-encoded string string to UTF8-encoded string
MRMESH_API std::u8string Utf16ToUtf8( const std::wstring_view & utf16 );
#endif

#if defined __cpp_lib_char8_t

inline const std::string & asString( const std::u8string & s ) { return reinterpret_cast<const std::string &>( s ); }
inline const std::u8string & asU8String( const std::string & s ) { return reinterpret_cast<const std::u8string &>( s ); }

inline std::string asString( std::u8string && s ) { return reinterpret_cast<std::string &&>( s ); }
inline std::u8string asU8String( std::string && s ) { return reinterpret_cast<std::u8string &&>( s ); }

inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::path( asU8String( std::string( s ) ) ); }

#else //std::u8string is not defined

inline const std::string & asString( const std::string & s ) { return s; }
inline const std::string & asU8String( const std::string & s ) { return s; }

inline std::string asString( std::string && s ) { return std::move( s ); }
inline std::string asU8String( std::string && s ) { return std::move( s ); }

inline std::filesystem::path pathFromUtf8( const char * s ) { return std::filesystem::u8path( s ); }

#endif

// returns filename as UTF8-encoded string
inline std::string utf8string( const std::filesystem::path & path )
    { return asString( path.u8string() ); }

}
