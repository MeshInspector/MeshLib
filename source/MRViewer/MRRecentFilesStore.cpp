#include "MRRecentFilesStore.h"
#include "MRMesh/MRConfig.h"
#include "MRMeshViewer.h"
#include "MRPch/MRSpdlog.h"

namespace
{
const std::string cRecentFilesStorageKey = "recentFileNames";
}

namespace MR
{

void RecentFilesStore::storeFile( const std::filesystem::path& file ) const
{
    if ( appName_.empty() )
    {
        spdlog::warn( "RecentFilesStore has no app name, data is not stored" );
        return;
    }
    auto& cfg = Config::instance();
    auto storedFiles = cfg.getFileStack( cRecentFilesStorageKey );
    auto inStorageIt = std::find( storedFiles.begin(), storedFiles.end(), file );
    if ( inStorageIt != storedFiles.end() )
        storedFiles.erase( inStorageIt );
    storedFiles.insert( storedFiles.begin(), file );
    if ( storedFiles.size() > capacity_ )
        storedFiles.resize( capacity_ );
    cfg.setFileStack( cRecentFilesStorageKey, storedFiles );
    storageUpdateSignal( storedFiles );
}

std::vector<std::filesystem::path> RecentFilesStore::getStoredFiles() const
{
    if ( appName_.empty() )
    {
        spdlog::warn( "RecentFilesStore has no app name, data is not read" );
        return {};
    }
    auto& cfg = Config::instance();
    return cfg.getFileStack( cRecentFilesStorageKey );
}

}
