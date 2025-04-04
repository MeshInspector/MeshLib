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

std::string getCurrentFolder( const std::filesystem::path& baseFolder )
{
    if ( !baseFolder.empty() )
        return utf8string( baseFolder );

    auto& cfg = Config::instance();
    if ( cfg.hasJsonValue( cLastUsedDirKey ) )
    {
        auto lastUsedDir = cfg.getJsonValue( cLastUsedDirKey );
        if ( lastUsedDir.isString() )
            return lastUsedDir.asString();
    }

    return utf8string( GetHomeDirectory() );
}

} // namespace MR::detail
