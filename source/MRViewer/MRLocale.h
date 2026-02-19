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
};

/// ...
inline auto gettext( const char* id )
{
    return boost::locale::gettext( id, Locale::get() );
}
/// ...
inline auto ngettext( const char* singular, const char* plural, auto n )
{
    return boost::locale::ngettext( singular, plural, n, Locale::get() );
}
/// ...
inline auto pgettext( const char* context, const char* id )
{
    return boost::locale::pgettext( context, id, Locale::get() );
}

} // namespace MR

#ifndef MR_NO_GETTEXT_MACROS
#define _tr( id ) MR::gettext( id ).c_str()
#define n_tr( s, p, n ) MR::ngettext( s, p, n ).c_str()
#define p_tr( ctx, id ) MR::pgettext( ctx, id ).c_str()
#endif // MR_NO_GETTEXT_MACROS
