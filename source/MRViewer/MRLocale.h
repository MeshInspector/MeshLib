#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_LOCALE
#include "exports.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR::Locale
{

/// ...
MRVIEWER_API const std::locale& get();
/// ...
MRVIEWER_API const std::string& getName();
/// ...
MRVIEWER_API const std::locale& set( std::string localeName );

/// ...
MRVIEWER_API std::vector<std::string> getAvailableLocales();

/// ...
MRVIEWER_API void addCatalogPath( const std::filesystem::path& path );
/// ...
MRVIEWER_API void addDomain( std::string domainName );

/// ...
MRVIEWER_API std::string getDisplayName( const std::string& localeName );
/// ...
MRVIEWER_API void setDisplayName( const std::string& localeName, const std::string& displayName );

/// ...
inline auto translate( const char* msg )
{
    return boost::locale::translate( msg ).str( get() );
}
/// ...
inline auto translate( const char* context, const char* msg )
{
    return boost::locale::translate( context, msg ).str( get() );
}
/// ...
inline auto translate( const char* single, const char* plural, auto n )
{
    return boost::locale::translate( single, plural, n ).str( get() );
}
/// ...
inline auto translate( const char* context, const char* single, const char* plural, auto n )
{
    return boost::locale::translate( context, single, plural, n ).str( get() );
}

} // namespace MR::Locale

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate( __VA_ARGS__ ).c_str()
#define f_tr( ... ) fmt::runtime( MR::Locale::translate( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#else // MRVIEWER_NO_LOCALE

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate_noop( __VA_ARGS__ )
#define f_tr( ... ) fmt::runtime( MR::Locale::translate_noop( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#endif // MRVIEWER_NO_LOCALE
