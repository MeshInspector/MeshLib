#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRSignal.h"
#include <string>
#include <vector>
#include <filesystem>

namespace MR
{

using FileNamesStack = std::vector<std::filesystem::path>;

/// This class exists to store and load recently opened files
/// from system storage (registry for Windows)
class RecentFilesStore
{
public:
    RecentFilesStore() = default;
    /// Capacity is maximum stored filenames,
    /// it needs app name to store data to correct place in system storage
    RecentFilesStore( std::string appName, int capacity = 10 ) :
        appName_{std::move(appName)}, capacity_{capacity} {}

    /// Saves filename on top of recently opened stack,
    /// if file is already in the storage put it on top
    MRVIEWER_API void storeFile( const std::filesystem::path& file ) const;

    /// Returns filenames from storage
    MRVIEWER_API std::vector<std::filesystem::path> getStoredFiles() const;

    /// Returns maximum size of recently opened files stack
    int getCapacity() const { return capacity_; }

    /// Connects given slot to receive signal on every update of this store
    MRVIEWER_API boost::signals2::connection onUpdate( const boost::function<void( const FileNamesStack& files )> & slot, boost::signals2::connect_position position = boost::signals2::at_back );

private:
    std::string appName_;
    int capacity_{5};
    boost::signals2::signal<void( const FileNamesStack& files )> updateSignal_;
};

} //namespace MR
