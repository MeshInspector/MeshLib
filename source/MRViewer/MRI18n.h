#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_LOCALE
#include "exports.h"

#include <MRMesh/MRMeshFwd.h>

#include <string>
#include <string_view>
#include <vector>

namespace MR::Locale
{

/// Information about locale domains. Contains the only field `id`.
/// The purpose of the struct is to resolve an ambiguous overload `translate( "str", "str", 0 )`
/// which otherwise could be treated as both "context, message, domain id" and "single, plural, number".
struct Domain
{
    int id = 0;
};

/// \brief Translates a message using the active locale.
MRVIEWER_API std::string translate( std::string_view msg, Domain domain = {} );

/// \brief Translates a message in context using the active locale.
MRVIEWER_API std::string translate( std::string_view context, std::string_view msg, Domain domain = {} );

/// \brief Translates a plural message form using the active locale.
MRVIEWER_API std::string translate( std::string_view single, std::string_view plural, Int64 n, Domain domain = {} );

/// \brief Translates a plural message form in context using the active locale.
MRVIEWER_API std::string translate( std::string_view context, std::string_view single, std::string_view plural, Int64 n, Domain domain = {} );

/// \brief Translates all strings in a vector using the active locale.
inline std::vector<std::string> translateAll( const std::vector<std::string>& items, Domain domain = {} )
{
    std::vector<std::string> result;
    result.reserve( items.size() );
    for ( const auto& s : items )
        result.push_back( MR::Locale::translate( s, domain ) );
    return result;
}

/// \brief Translates all strings in a vector with context using the active locale.
inline std::vector<std::string> translateAll( const char* context, const std::vector<std::string>& items, Domain domain = {} )
{
    std::vector<std::string> result;
    result.reserve( items.size() );
    for ( const auto& s : items )
        result.push_back( MR::Locale::translate( context, s, domain ) );
    return result;
}

} // namespace MR::Locale

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate( __VA_ARGS__ ).c_str()
#define s_tr( ... ) MR::Locale::translate( __VA_ARGS__ )
#define f_tr( ... ) fmt::runtime( MR::Locale::translate( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#else // MRVIEWER_NO_LOCALE

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate_noop( __VA_ARGS__ )
#define s_tr( ... ) std::string( MR::Locale::translate_noop( __VA_ARGS__ ) )
#define f_tr( ... ) fmt::runtime( MR::Locale::translate_noop( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#endif // MRVIEWER_NO_LOCALE
