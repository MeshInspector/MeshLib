#pragma once

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR
{

/// ...
class Locale
{
public:
    /// ...
    static void init();
    /// ...
    static const std::locale& get();
    /// ...
    static const char* getName();
    /// ...
    static const std::locale& set( const char* locale );
    /// ...
    static std::vector<std::string> getAvailableLocales();

    /// ...
    static std::string getDisplayName( const char* locale );
    /// ...
    static void addDisplayName( const char* locale, const std::string& name );

    /// ...
    static auto translate( const char* msg )
    {
        return boost::locale::translate( msg ).str( get() );
    }
    /// ...
    static auto translate( const char* context, const char* msg )
    {
        return boost::locale::translate( context, msg ).str( get() );
    }
    /// ...
    static auto translate( const char* single, const char* plural, auto n )
    {
        return boost::locale::translate( single, plural, n ).str( get() );
    }
    /// ...
    static auto translate( const char* context, const char* single, const char* plural, auto n )
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
