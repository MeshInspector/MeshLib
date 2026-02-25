#include "MRLocale.h"

#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/generator.hpp>
#pragma warning( pop )

namespace MR
{

namespace
{

// enforce xgettext to use UTF-8 encoding
// TRANSLATORS: this is a technical string; no need to translate it
[[maybe_unused]] constexpr auto cEnforceUtf8 = _t( "(^̮^)" );

std::locale gLocale = {};
std::string gLocaleName = "en";
boost::locale::generator gLocaleGen = {};

std::filesystem::path localeDir()
{
    return SystemPath::getResourcesDirectory() / "locale";
}

std::unordered_map<std::string, std::string> gKnownLocales = {
#include "MRLocaleNames.inl"
};

} // namespace

void Locale::init()
{
    gLocaleGen.add_messages_path( utf8string( localeDir() ) );
    gLocaleGen.add_messages_domain( MR_PROJECT_NAME "/utf-8" );
    gLocaleGen.categories( boost::locale::category_t::message );
    // do not generate wide-character facets (this is crucial for Wasm support as we don't use ICU or iconv)
    gLocaleGen.characters( boost::locale::char_facet_t::char_f );
    gLocale = gLocaleGen.generate( gLocaleName );
}

const std::locale& Locale::get()
{
    return gLocale;
}

const char* Locale::getName()
{
    return gLocaleName.c_str();
}

const std::locale& Locale::set( const char* locale )
{
    auto localeName = ( gLocaleName = locale );
    // TODO: correct encoding processing
    if ( !localeName.ends_with( ".UTF-8" ) )
        localeName.append( ".UTF-8" );
    return ( gLocale = gLocaleGen.generate( localeName ) );
}

std::vector<std::string> Locale::getAvailableLocales()
{
    std::vector<std::string> results;
    // always add English
    results.emplace_back( "en" );
    std::error_code ec;
    for ( const auto& entry : Directory{ localeDir(), ec } )
        if ( entry.is_directory( ec ) )
            results.emplace_back( utf8string( entry.path().filename() ) );
    std::sort( results.begin(), results.end() );
    return results;
}

std::string Locale::getDisplayName( const char* locale )
{
    const auto it = gKnownLocales.find( locale );
    if ( it != gKnownLocales.end() )
        return it->second;
    return locale;
}

void Locale::addDisplayName( const char* locale, const std::string& name )
{
    gKnownLocales[locale] = name;
}

} // namespace MR
