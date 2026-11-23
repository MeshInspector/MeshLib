#include "MRLocale.h"
#ifndef MRVIEWER_NO_LOCALE
#include "MRTranslationFormat.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRTelemetry.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/generator.hpp>
#include <boost/locale/gnu_gettext.hpp>
#include <boost/locale/info.hpp>
#include <boost/locale/message.hpp>
#include <boost/locale/util.hpp>
#pragma warning( pop )
#include <boost/signals2/signal.hpp>
#include <boost/version.hpp>

#ifdef _WIN32
#include <WinNls.h>
#endif

#ifdef __APPLE__
#include "MRLocaleMacos.h"
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string_view>
#include <unordered_set>

namespace MR
{

namespace
{

std::locale gLocale = {};
std::string gLocaleName = "en";
std::string gLocaleCanonicalName = "en.UTF-8";
boost::signals2::signal<void ( const std::string& )> gLocaleNameChanged = {};
boost::locale::generator gLocaleGen = {};
std::vector<std::string> gLocaleDomains = {};
std::vector<std::filesystem::path> gLocaleDirs = {};
bool gCheckTranslations = false;
std::vector<LocaleDomainId> gGenericDomains = {};

std::filesystem::path getLocaleDir()
{
    return SystemPath::getResourcesDirectory() / "locale";
}

std::unordered_map<std::string, std::string> gKnownLocales = {
#include "MRLocaleNames.inl"
};

std::map<const char*, LocaleDomainId> gDomainCache = {};

std::uint32_t swapBytes( std::uint32_t v )
{
    return ( v >> 24 ) | ( ( v >> 8 ) & 0xff00 ) | ( ( v << 8 ) & 0xff0000 ) | ( v << 24 );
}

/// makes not findable in the contents of .mo file the translations that cannot replace their originals in fmt::format or printf call
void disableUnsafeTranslations( std::vector<char>& file, const std::string& fileName )
{
    constexpr std::uint32_t cMagic = 0x950412de;
    if ( file.size() < 28 ) // header
        return;
    bool swapped = false; // the file was written with another byte order
    auto get = [&]( size_t offset )
    {
        std::uint32_t v;
        std::memcpy( &v, file.data() + offset, sizeof( v ) );
        return swapped ? swapBytes( v ) : v;
    };
    if ( get( 0 ) != cMagic )
    {
        swapped = true;
        if ( get( 0 ) != cMagic )
            return;
    }
    // tables of (length, offset) of original and translated strings
    const std::uint64_t num = get( 8 ), tables[2] = { get( 12 ), get( 16 ) };
    if ( tables[0] + num * 8 > file.size() || tables[1] + num * 8 > file.size() )
        return; // an invalid file is rejected by boost

    for ( std::uint64_t i = 0; i < num; ++i )
    {
        std::string_view entry[2];
        for ( int k = 0; k < 2; ++k )
        {
            const auto record = size_t( tables[k] + i * 8 );
            const std::uint64_t length = get( record ), offset = get( record + 4 );
            if ( offset + length >= file.size() )
                return;
            entry[k] = std::string_view( file.data() + size_t( offset ), size_t( length ) );
        }
        // the key is [context EOT] original [NUL plural original], the value is one or several forms separated by NUL
        auto original = entry[0].substr( entry[0].find( '\x04' ) + 1 );
        std::string_view plural;
        if ( const auto pos = original.find( '\0' ); pos != std::string_view::npos )
        {
            plural = original.substr( pos + 1 );
            original = original.substr( 0, pos );
        }
        bool safe = true;
        // the entry with empty original is the header of the catalog
        for ( auto forms = entry[1]; safe && !original.empty(); )
        {
            const auto pos = forms.find( '\0' );
            const auto form = forms.substr( 0, pos );
            safe = Locale::isFormatSafeTranslation( original, form ) || ( !plural.empty() && Locale::isFormatSafeTranslation( plural, form ) );
            if ( !safe )
            {
                // reported once, although the catalogs are loaded again with every new domain
                static std::unordered_set<std::string> reported;
                std::string id( fileName );
                id.push_back( '\n' );
                id.append( entry[0] );
                if ( reported.insert( std::move( id ) ).second )
                    spdlog::warn( "{}: translation \"{}\" of \"{}\" is ignored: its format placeholders do not match the original", fileName, form, original );
            }
            if ( pos == std::string_view::npos )
                break;
            forms = forms.substr( pos + 1 );
        }
        // no string to translate starts with this character, and the hash table of the file stays valid
        if ( !safe )
            file[size_t( entry[0].data() - file.data() )] = '\x01';
    }
}

/// reads .mo file for boost, returns empty vector if there is no such file
std::vector<char> loadCheckedCatalog( const std::string& fileName, const std::string& )
{
    std::ifstream in( pathFromUtf8( fileName ), std::ifstream::binary | std::ifstream::ate );
    if ( !in )
        return {};
    const auto size = std::streamoff( in.tellg() );
    if ( size <= 0 )
        return {};
    std::vector<char> res( ( size_t( size ) ) );
    in.seekg( 0 );
    if ( !in.read( res.data(), size ) )
        return {};
    disableUnsafeTranslations( res, fileName );
    return res;
}

/// makes the active locale again after a change of its name, domains, catalog paths or the way of loading
void updateLocale()
{
    if ( !gCheckTranslations )
    {
        gLocale = gLocaleGen.generate( gLocaleCanonicalName );
        return;
    }
    // the same message facet as the generator makes, but with the files read by our function
    const auto infoLocale = boost::locale::util::create_info( std::locale::classic(), gLocaleCanonicalName );
    const auto& data = std::use_facet<boost::locale::info>( infoLocale );
    boost::locale::gnu_gettext::messages_info info;
    info.language = data.language();
    info.country = data.country();
    info.variant = data.variant();
    info.encoding = data.encoding();
    for ( const auto& domain : gLocaleDomains )
        info.domains.emplace_back( domain );
    for ( const auto& dir : gLocaleDirs )
        info.paths.push_back( utf8string( dir ) );
    info.callback = loadCheckedCatalog;
    gLocale = std::locale( std::locale::classic(), boost::locale::gnu_gettext::create_messages_facet<char>( info ) );
}

/// adds new domain and updates the active locale
void addLocaleDomain( const std::string& domainName )
{
    if ( std::find( gLocaleDomains.begin(), gLocaleDomains.end(), domainName ) == gLocaleDomains.end() )
        gLocaleDomains.push_back( domainName );
    gLocaleGen.add_messages_domain( domainName );
    updateLocale();
}

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
    updateLocale();
    return gLocale;
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
    updateLocale();
}

LocaleDomainId Locale::addDomain( const char* domainName, bool generic )
{
    if ( auto it = gDomainCache.find( domainName ); it != gDomainCache.end() )
        return it->second;

    addLocaleDomain( domainName );

    const auto id = findDomain( std::string{ domainName } );
    if ( id )
    {
        gDomainCache[domainName] = id;
        if ( generic )
            gGenericDomains.emplace_back( id );
    }
    return id;
}

LocaleDomainId Locale::addDomain( const std::string& domainName, bool generic )
{
    addLocaleDomain( domainName );

    const auto id = findDomain( domainName );
    if ( id && generic )
        gGenericDomains.emplace_back( id );
    return id;
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

const std::vector<LocaleDomainId>& Locale::getGenericDomains()
{
    return gGenericDomains;
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

void Locale::enableTranslationCheck( bool on )
{
    if ( gCheckTranslations == on )
        return;
    gCheckTranslations = on;
    updateLocale();
}

bool Locale::isTranslationCheckEnabled()
{
    return gCheckTranslations;
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
            [[maybe_unused]] ULONG actualNumLangs = 0;
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
    // explicitly add MeshLib domain to generic domains
    gGenericDomains.emplace_back( 0 );
};

} // namespace MR
#endif
