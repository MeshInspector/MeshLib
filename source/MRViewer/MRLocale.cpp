#include "MRLocale.h"

#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRTelemetry.h"
#include "MRPch/MRWasm.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/generator.hpp>
#include <boost/locale/message.hpp>
#pragma warning( pop )
#include <boost/signals2/signal.hpp>
#include <boost/version.hpp>

#ifdef _WIN32
#include <WinNls.h>
#endif

#ifdef __APPLE__
#include "MRLocaleMacos.h"
#endif

#include <cassert>
#include <map>

namespace MR
{

namespace
{

std::locale gLocale = {};
std::string gLocaleName = "en";
std::string gLocaleCanonicalName = "en.UTF-8";
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

std::map<const char*, LocaleDomainId> gDomainCache = {};

} // namespace

const std::locale& Locale::get()
{
    return gLocale;
}

const std::string& Locale::getName()
{
    return gLocaleName;
}

const std::locale& Locale::set( const std::string& localeName )
{
    gLocaleName = localeName;
    MR_FINALLY {
        gLocaleNameChanged( gLocaleName );
        TelemetrySignal( "Set Language " + gLocaleName );
    };

    gLocaleCanonicalName = localeName;
    // TODO: correct encoding processing
    if ( !gLocaleCanonicalName.ends_with( ".UTF-8" ) )
        gLocaleCanonicalName.append( ".UTF-8" );
    return ( gLocale = gLocaleGen.generate( gLocaleCanonicalName ) );
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
    results.erase( std::unique( results.begin(), results.end() ), results.end() );
    return results;
}

void Locale::addCatalogPath( const std::filesystem::path& path )
{
    if ( std::find( gLocaleDirs.begin(), gLocaleDirs.end(), path ) != gLocaleDirs.end() )
        return;
    gLocaleDirs.emplace_back( path );
    gLocaleGen.add_messages_path( utf8string( path ) );
    gLocale = gLocaleGen.generate( gLocaleCanonicalName );
}

LocaleDomainId Locale::addDomain( const char* domainName )
{
    if ( auto it = gDomainCache.find( domainName ); it != gDomainCache.end() )
        return it->second;

    gLocaleGen.add_messages_domain( domainName );
    gLocale = gLocaleGen.generate( gLocaleCanonicalName );

    const auto id = findDomain( std::string{ domainName } );
    if ( id )
        gDomainCache[domainName] = id;
    return id;
}

LocaleDomainId Locale::addDomain( const std::string& domainName )
{
    gLocaleGen.add_messages_domain( domainName );
    gLocale = gLocaleGen.generate( gLocaleCanonicalName );

    return findDomain( domainName );
}

LocaleDomainId Locale::findDomain( const char* domainName )
{
    if ( auto it = gDomainCache.find( domainName ); it != gDomainCache.end() )
        return it->second;

    const auto id = findDomain( std::string{ domainName } );
    if ( id )
        gDomainCache[domainName] = id;
    return id;
}

LocaleDomainId Locale::findDomain( const std::string& domainName )
{
    using facet_type = boost::locale::message_format<char>;
    assert( std::has_facet<facet_type>( gLocale ) );
    const auto id = std::use_facet<facet_type>( gLocale ).domain( domainName );
    return id != 0 ? LocaleDomainId{ id } : LocaleDomainId{};
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

std::vector<std::string> Locale::getSystemLocales()
{
    [[maybe_unused]] constexpr auto fromBcp47 = [] ( std::string langName )
    {
        std::replace( langName.begin(), langName.end(), '-', '_' );
        // see also: https://learn.microsoft.com/en-us/windows/win32/intl/locale-names
        return langName;
    };
    [[maybe_unused]] constexpr auto fromPosix = [] ( std::string langName )
    {
        // strip encoding
        if ( auto pos = langName.find( '.' ); pos != std::string::npos )
            langName = langName.substr( 0, pos );
        // strip variant
        if ( auto pos = langName.find( '@' ); pos != std::string::npos )
            langName = langName.substr( 0, pos );
        return langName;
    };

    std::vector<std::string> results;
    auto addLang = [&] ( const std::string& langName )
    {
        results.emplace_back( langName );
        // also add the language code alone (de_CH -> de)
        if ( auto pos = langName.find( '_' ); pos != std::string::npos )
            results.emplace_back( langName.substr( 0, pos ) );
    };
#if defined _WIN32
    ULONG numLangs, bufSize = 0;
    if ( TRUE == GetUserPreferredUILanguages( MUI_LANGUAGE_NAME, &numLangs, NULL, &bufSize ) )
    {
        std::wstring buf( bufSize, L'\0' );
        if ( TRUE == GetUserPreferredUILanguages( MUI_LANGUAGE_NAME, &numLangs, buf.data(), &bufSize ) )
        {
            ULONG actualNumLangs = 0;
            for ( auto ptr = buf.data(); ptr[0] != L'\0'; ptr += std::wcslen( ptr ) + 1 )
            {
                actualNumLangs++;
                const auto langName = wideToUtf8( ptr );
                addLang( fromBcp47( langName ) );
            }
            assert( actualNumLangs == numLangs );
        }
    }
#elif defined __APPLE__
    for ( auto langName : detail::getMacosLocales() )
        addLang( fromBcp47( langName ) );
#elif defined __EMSCRIPTEN__
    auto* langNames = (char*)EM_ASM_PTR( { return stringToNewUTF8( navigator.languages.join() ); } );
    for ( auto langName : split( langNames, "," ) )
        addLang( fromBcp47( langName ) );
    free( langNames );
#else
    constexpr auto getenv = [] ( const char* name ) -> char*
    {
        if ( auto* var = std::getenv( name ); var != nullptr && *var != '\0' )
            return var;
        return nullptr;
    };

    // https://www.gnu.org/software/gettext/manual/html_node/Locale-Environment-Variables.html
    // https://www.gnu.org/software/gettext/manual/html_node/The-LANGUAGE-variable.html
    std::string currentLocale = "C";
    if ( const auto* envLcAll = getenv( "LC_ALL" ) )
        currentLocale = fromPosix( envLcAll );
    else if ( const auto* envLcMessages = getenv( "LC_MESSAGES" ) )
        currentLocale = fromPosix( envLcMessages );
    else if ( const auto* envLang = getenv( "LANG" ) )
        currentLocale = fromPosix( envLang );
    if ( currentLocale != "C" )
    {
        addLang( currentLocale );

        if ( const auto* envLanguage = getenv( "LANGUAGE" ) )
            for ( auto langName : split( envLanguage, ":" ) )
                addLang( fromPosix( langName ) );
    }
#endif

    // remove duplicates
    std::set<std::string> langSet;
    auto it = std::remove_if( results.begin(), results.end(), [&] ( const std::string& langName )
    {
        auto [_, inserted] = langSet.insert( langName );
        return !inserted;
    } );
    results.erase( it, results.end() );

    return results;
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
