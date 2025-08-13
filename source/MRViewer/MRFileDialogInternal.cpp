#include "MRFileDialogInternal.h"

#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"

namespace
{

const std::string cLastUsedDirKey = "lastUsedDir";

} // namespace

namespace MR::detail
{

std::string getLastUsedDir()
{
    auto& cfg = Config::instance();
    if ( cfg.hasJsonValue( cLastUsedDirKey ) )
    {
        auto lastUsedDir = cfg.getJsonValue( cLastUsedDirKey );
        if ( lastUsedDir.isString() )
            return lastUsedDir.asString();
    }

    return utf8string( GetHomeDirectory() );
}

void setLastUsedDir( const std::string& folder )
{
    auto& cfg = Config::instance();
    cfg.setJsonValue( cLastUsedDirKey, folder );
}

} // namespace MR::detail
