#include "MRLocale.h"

#include "MRMesh/MRDirectory.h"
#include "MRMesh/MROnInit.h"
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
std::vector<std::filesystem::path> gLocaleDirs = {};

std::filesystem::path getLocaleDir()
{
    return SystemPath::getResourcesDirectory() / "locale";
}

std::unordered_map<std::string, std::string> gKnownLocales = {
#include "MRLocaleNames.inl"
};

} // namespace

const std::locale& Locale::get()
{
    return gLocale;
}

const std::string& Locale::getName()
{
    return gLocaleName;
}

const std::locale& Locale::set( std::string localeName )
{
    gLocaleName = localeName;

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
    for ( const auto& dir : gLocaleDirs )
        for ( const auto& entry : Directory{ dir, ec } )
            if ( entry.is_directory( ec ) )
                results.emplace_back( utf8string( entry.path().filename() ) );
    std::sort( results.begin(), results.end() );
    return results;
}

void Locale::addCatalogPath( const std::filesystem::path& path )
{
    if ( std::find( gLocaleDirs.begin(), gLocaleDirs.end(), path ) != gLocaleDirs.end() )
        return;
    gLocaleDirs.emplace_back( path );
    gLocaleGen.add_messages_path( utf8string( path ) );
    gLocale = gLocaleGen.generate( gLocaleName );
}

void Locale::addDomain( std::string domainName )
{
    // force set UTF-8 encoding
    if ( auto pos = domainName.find( '/' ); pos != std::string::npos )
        domainName.replace( pos, domainName.size(), "/utf-8" );
    else
        domainName.append( "/utf-8" );

    gLocaleGen.add_messages_domain( domainName );
    gLocale = gLocaleGen.generate( gLocaleName );
}

std::string Locale::getDisplayName( const std::string& localeName )
{
    const auto it = gKnownLocales.find( localeName );
    if ( it != gKnownLocales.end() )
        return it->second;
    return localeName;
}

void Locale::setDisplayName( const std::string& localeName, const std::string& displayName )
{
    gKnownLocales[localeName] = displayName;
}

MR_ON_INIT
{
    // generate message facets only
    gLocaleGen.categories( boost::locale::category_t::message );
    // do not generate wide-character facets (this is crucial for Wasm support as we don't use ICU or iconv)
    gLocaleGen.characters( boost::locale::char_facet_t::char_f );
    // set MeshLib domain by default
    gLocaleGen.add_messages_domain( "MeshLib/utf-8" );
    Locale::addCatalogPath( getLocaleDir() );
};

} // namespace MR
