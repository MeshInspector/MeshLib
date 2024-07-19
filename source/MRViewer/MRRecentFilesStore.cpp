#include "MRRecentFilesStore.h"
#include "MRMesh/MRConfig.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"

namespace
{
const std::string cRecentFilesStorageKey = "recentFileNames";
}

namespace MR
{

#ifdef __EMSCRIPTEN__
void RecentFilesStore::storeFile( const std::filesystem::path& ) const
{}
#else
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
    updateSignal_( storedFiles );
}
#endif

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

boost::signals2::connection RecentFilesStore::onUpdate( const boost::function<void( const FileNamesStack& files )> & slot, boost::signals2::connect_position position )
{
    return updateSignal_.connect( slot, position );
}

} //namespace MR
