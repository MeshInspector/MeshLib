#include "MRLocale.h"

#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/generator.hpp>
#include <boost/locale/message.hpp>
#pragma warning( pop )
#include <boost/signals2/signal.hpp>
#include <boost/version.hpp>

#include <cassert>
#include <map>

namespace MR
{

namespace
{

std::locale gLocale = {};
std::string gLocaleName = "en";
boost::signals2::signal<void ( const std::string& )> gLocaleNameChanged = {};
boost::locale::generator gLocaleGen = {};
std::vector<std::filesystem::path> gLocaleDirs = {};

std::filesystem::path getLocaleDir()
{
    return SystemPath::getResourcesDirectory() / "locale";
}

std::unordered_map<std::string, std::string> gKnownLocales = {
#include "MRLocaleNames.inl"
};

std::map<const char*, int> gDomainCache = {};

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
    MR_FINALLY {
        gLocaleNameChanged( gLocaleName );
    };

    // TODO: correct encoding processing
    if ( !localeName.ends_with( ".UTF-8" ) )
        localeName.append( ".UTF-8" );
    return ( gLocale = gLocaleGen.generate( localeName ) );
}

boost::signals2::connection Locale::onChanged( const std::function<void ( const std::string& )>& cb )
{
    return gLocaleNameChanged.connect( cb );
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

int Locale::addDomain( const char* domainName )
{
    if ( auto it = gDomainCache.find( domainName ); it != gDomainCache.end() )
        return it->second;

    gLocaleGen.add_messages_domain( domainName );
    gLocale = gLocaleGen.generate( gLocaleName );

    using facet_type = boost::locale::message_format<char>;
    assert( std::has_facet<facet_type>( gLocale ) );
    return ( gDomainCache[domainName] = std::use_facet<facet_type>( gLocale ).domain( domainName ) );
}

int Locale::addDomain( const std::string& domainName )
{
    gLocaleGen.add_messages_domain( domainName );
    gLocale = gLocaleGen.generate( gLocaleName );

    return findDomain( domainName );
}

int Locale::findDomain( const char* domainName )
{
    if ( auto it = gDomainCache.find( domainName ); it != gDomainCache.end() )
        return it->second;

    using facet_type = boost::locale::message_format<char>;
    assert( std::has_facet<facet_type>( gLocale ) );
    return ( gDomainCache[domainName] = std::use_facet<facet_type>( gLocale ).domain( domainName ) );
}

int Locale::findDomain( const std::string& domainName )
{
    using facet_type = boost::locale::message_format<char>;
    assert( std::has_facet<facet_type>( gLocale ) );
    return std::use_facet<facet_type>( gLocale ).domain( domainName );
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
    gLocaleGen.categories(
#if BOOST_VERSION >= 108100
        boost::locale::category_t::message
#else
        boost::locale::message_facet
#endif
    );
    // do not generate wide-character facets (this is crucial for Wasm support as we don't use ICU or iconv)
    gLocaleGen.characters(
#if BOOST_VERSION >= 108100
        boost::locale::char_facet_t::char_f
#else
        boost::locale::char_facet
#endif
    );
    // set MeshLib domain by default
    Locale::addDomain( "MeshLib" );
    Locale::addCatalogPath( getLocaleDir() );
};

} // namespace MR
