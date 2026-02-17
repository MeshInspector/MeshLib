#include "MRLocale.h"

#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"

#include <boost/locale/generator.hpp>

namespace MR
{

namespace
{

std::locale gLocale = {};
std::string gLocaleName = "en";
boost::locale::generator gLocaleGen = {};

std::filesystem::path localeDir()
{
    return SystemPath::getResourcesDirectory() / "locale";
}

} // namespace

void Locale::init()
{
    gLocaleGen.add_messages_path( utf8string( localeDir() ) );
    gLocaleGen.add_messages_domain( MR_PROJECT_NAME "/utf-8" );
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

} // namespace MR
