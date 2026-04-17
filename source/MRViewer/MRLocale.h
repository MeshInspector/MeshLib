#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_LOCALE
#include "exports.h"

#include "MRMesh/MRMeshFwd.h"

#include <boost/signals2/connection.hpp>

#include <filesystem>
#include <vector>

namespace MR::Locale
{

/// \brief Returns the active locale.
MRVIEWER_API const std::locale& get();
/// \brief Returns the active locale's name.
/// \note If the locale was set manually, this function will return its name as is, without any normalization.
MRVIEWER_API const std::string& getName();
/// \brief Loads and sets the active locale by its name. UTF-8 is always used as an encoding.
/// \returns Reference to the new locale.
MRVIEWER_API const std::locale& set( const std::string& localeName );
/// \brief Connects to a signal emitted every time the active locale is changed.
MRVIEWER_API boost::signals2::connection onChanged( const std::function<void ( const std::string& )>& cb );

/// \brief Returns the list of the names of locales with available .mo files.
/// "en" is always included as the default locale.
/// The returned list is always sorted alphabetically.
MRVIEWER_API std::vector<std::string> getAvailableLocales();

/// \brief Adds a path to .mo files.
/// The path is expected to have the following directory structure:
///   <locale name>/LC_MESSAGES/<domain name>.mo
/// The active locale is reloaded on every call.
MRVIEWER_API void addCatalogPath( const std::filesystem::path& path );
/// \brief Adds a new domain.
/// The active locale is reloaded on every call.
/// \note This overload is meant to be used with string literals for faster lookups.
/// \returns The id of the added domain.
MRVIEWER_API LocaleDomainId addDomain( const char* domainName );
/// \brief Adds a new domain.
/// The active locale is reloaded on every call.
/// \returns The id of the added domain.
MRVIEWER_API LocaleDomainId addDomain( const std::string& domainName );
/// \brief Find an id for the given domain that can be passed to the `translate` functions.
/// \returns The domain id if the domain is previously added and invalid id otherwise.
/// \note This overload is meant to be used with string literals for faster lookups.
/// \ref translate
MRVIEWER_API LocaleDomainId findDomain( const char* domainName );
/// \brief Find an id for the given domain that can be passed to the `translate` functions.
/// \returns The domain id if the domain is previously added and invalid id otherwise.
/// \ref translate
MRVIEWER_API LocaleDomainId findDomain( const std::string& domainName );

/// \brief Returns a display name for the given locale.
/// \returns
///  - The display name explicitly set with setDisplayName;
///  - or the display name from the compiled-in list of names generated from CLDR data;
///  - or the locale name as is.
/// \ref setDisplayName
MRVIEWER_API std::string getDisplayName( const std::string& localeName );
/// \brief Adds or updates a display name for the given locale.
MRVIEWER_API void setDisplayName( const std::string& localeName, const std::string& displayName );

/// \brief Returns a list of system locales.
/// The first one in the list is always the active system locale.
MRVIEWER_API std::vector<std::string> getSystemLocales();

} // namespace MR::Locale
#endif // MRVIEWER_NO_LOCALE
