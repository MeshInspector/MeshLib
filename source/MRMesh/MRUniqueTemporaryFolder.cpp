#include "MRUniqueTemporaryFolder.h"

#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

UniqueTemporaryFolder::UniqueTemporaryFolder( FolderCallback onPreTempFolderDelete )
    : onPreTempFolderDelete_( std::move( onPreTempFolderDelete ) )
{
    MR_TIMER

    std::error_code ec;
    const auto tmp = std::filesystem::temp_directory_path( ec );
    if ( ec )
    {
        spdlog::error( "Cannot get temporary directory: {}", systemToUtf8( ec.message() ) );
        return;
    }

    constexpr int MAX_ATTEMPTS = 32;
    // if the process is terminated in between temporary folder creation and removal, then
    // all 32 folders can be present, so we use current time to ignore old folders
    auto t0 = std::time( nullptr );
    for ( int i = 0; i < MAX_ATTEMPTS; ++i )
    {
        auto folder = tmp / ( "MeshInspectorScene" + std::to_string( t0 + i ) );
        if ( create_directories( folder, ec ) )
        {
            folder_ = std::move( folder );
            spdlog::info( "Temporary folder created: {}", utf8string( folder_ ) );
            break;
        }
    }
    if ( folder_.empty() )
        spdlog::error( "Failed to create unique temporary folder" );
}

UniqueTemporaryFolder::~UniqueTemporaryFolder()
{
    if ( folder_.empty() )
        return;

    MR_TIMER

    if ( onPreTempFolderDelete_ )
        onPreTempFolderDelete_( folder_ );

    spdlog::info( "Deleting temporary folder: {}", utf8string( folder_ ) );
    std::error_code ec;
    if ( !std::filesystem::remove_all( folder_, ec ) )
        // result may be zero if folder_ did not exist to begin with, see https://en.cppreference.com/w/cpp/filesystem/remove
        spdlog::error( "Folder {} did not exist", utf8string( folder_ ) );
    else if ( ec )
        spdlog::error( "Deleting folder {} failed: {}", utf8string( folder_ ), systemToUtf8( ec.message() ) );
}

} // namespace MR
