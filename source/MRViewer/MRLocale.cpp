#include "MRLocale.h"

#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"

#include <boost/locale/generator.hpp>

namespace MR
{

namespace
{

std::locale gLocale = {};
boost::locale::generator gLocaleGen = {};

} // namespace

void Locale::init()
{
    gLocaleGen.add_messages_path( utf8string( SystemPath::getResourcesDirectory() / "locale" ) );
    gLocaleGen.add_messages_domain( MR_PROJECT_NAME "/utf-8" );
}

const std::locale& Locale::get()
{
    return gLocale;
}

const std::locale& Locale::set( const char* locale )
{
    return ( gLocale = gLocaleGen.generate( locale ) );
}

} // namespace MR
