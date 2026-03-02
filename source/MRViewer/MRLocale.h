#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_LOCALE
#include "exports.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR
{

namespace Locale
{
    /// ...
    MRVIEWER_API void init();
    /// ...
    MRVIEWER_API const std::locale& get();
    /// ...
    MRVIEWER_API const char* getName();
    /// ...
    MRVIEWER_API const std::locale& set( const char* locale );
    /// ...
    MRVIEWER_API std::vector<std::string> getAvailableLocales();

    /// ...
    MRVIEWER_API std::string getDisplayName( const char* locale );
    /// ...
    MRVIEWER_API void addDisplayName( const char* locale, const std::string& name );

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
};

/// ...
inline auto gettext( const char* msg )
{
    return boost::locale::gettext( msg, Locale::get() );
}
/// ...
inline auto pgettext( const char* context, const char* msg )
{
    return boost::locale::pgettext( context, msg, Locale::get() );
}
/// ...
inline auto ngettext( const char* single, const char* plural, auto n )
{
    return boost::locale::ngettext( single, plural, n, Locale::get() );
}
/// ...
inline auto npgettext( const char* context, const char* single, const char* plural, auto n )
{
    return boost::locale::npgettext( context, single, plural, n, Locale::get() );
}

} // namespace MR

#ifndef MR_NO_GETTEXT_MACROS
#define _tr( ... ) MR::Locale::translate( __VA_ARGS__ ).c_str()
#endif // MR_NO_GETTEXT_MACROS

#endif // MRVIEWER_NO_LOCALE

