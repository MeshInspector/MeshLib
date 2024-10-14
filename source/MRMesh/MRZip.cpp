#include "MRZip.h"
#include "MRDirectory.h"
#include "MRIOParsing.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#endif

#include <zip.h>
#include <zipconf.h>

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

#include <cassert>
#include <fstream>

namespace MR
{

namespace {

struct ProgressData
{
    ProgressCallback cb = nullptr;
    bool canceled = false;
};

void zipProgressCallback( zip_t* , double progress, void* data )
{
    if ( !data )
        return;
    
    auto pd = reinterpret_cast<ProgressData*>( data );
    if ( !reportProgress( pd->cb, float( progress ) ) )
        pd->canceled = true;
}

#if (defined(LIBZIP_VERSION_MINOR) && LIBZIP_VERSION_MINOR >= 6 )

int zipCancelCallback( zip_t* , void* data )
{
    if ( !data )
        return 0;

    auto pd = reinterpret_cast< ProgressData* >( data );
    return int( pd->canceled );
}

#endif

// this object stores a handle on open zip-archive, and automatically closes it in the destructor
class AutoCloseZip
{
public:
    AutoCloseZip( const char* path, int flags, int* err, ProgressCallback cb = nullptr )
    {
        handle_ = zip_open( path, flags, err );
        pd_.cb = cb;
    }
    AutoCloseZip( zip_source_t & source, int flags, zip_error_t* err, ProgressCallback cb = nullptr )
    {
        handle_ = zip_open_from_source( &source, flags, err );
        pd_.cb = cb;
    }
    ~AutoCloseZip()
    {
        close();
    }
    operator zip_t *() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }
    int close()
    {
        if ( !handle_ )
            return 0;
        zip_register_progress_callback_with_state( handle_, 0.001f, zipProgressCallback, nullptr, &pd_ );
#if (defined(LIBZIP_VERSION_MINOR) && LIBZIP_VERSION_MINOR >= 6 )
        zip_register_cancel_callback_with_state( handle_, zipCancelCallback, nullptr, &pd_ );
#endif
        int res = zip_close( handle_ );
        handle_ = nullptr;
        return res;
    }

private:
    zip_t * handle_ = nullptr;
    ProgressData pd_;
};

/// zip-callback for reading from std::istream
zip_int64_t istreamZipSourceCallback( void *istream, void *data, zip_uint64_t len, zip_source_cmd_t cmd )
{
    if ( !istream )
    {
        assert( false );
        return -1;
    }

    std::istream & is = *(std::istream*)( istream );

    switch ( cmd )
    {
        case ZIP_SOURCE_SUPPORTS:
            return zip_source_make_command_bitmap( ZIP_SOURCE_OPEN, ZIP_SOURCE_READ, ZIP_SOURCE_CLOSE,
                ZIP_SOURCE_STAT, ZIP_SOURCE_ERROR, ZIP_SOURCE_FREE, ZIP_SOURCE_SEEK, ZIP_SOURCE_TELL, ZIP_SOURCE_SUPPORTS, -1 );

        case ZIP_SOURCE_SEEK:
        {
            zip_source_args_seek argsSeek = * ((zip_source_args_seek *)data);
            if ( argsSeek.whence == SEEK_SET && argsSeek.offset >= 0 )
                is.seekg( argsSeek.offset );
            else if ( argsSeek.whence == SEEK_CUR )
                is.seekg( argsSeek.offset, std::ios_base::cur );
            else if ( argsSeek.whence == SEEK_END && argsSeek.offset <= 0 )
                is.seekg( argsSeek.offset, std::ios_base::end );
            else
            {
                assert( false );
                return -1;
            }
            assert( !is.fail() );
            return is.fail() ? -1 : 0;
        }

        case ZIP_SOURCE_OPEN:
            return 0;

        case ZIP_SOURCE_READ:
            is.read( (char*)data, len );
            assert( !is.fail() );
            return is.fail() ? 0 : len;

        case ZIP_SOURCE_CLOSE:
            return 0;

        case ZIP_SOURCE_TELL:
            return is.tellg();

        case ZIP_SOURCE_STAT:
        {
            zip_stat_t* zipStat = (zip_stat_t*)data;
            zip_stat_init(zipStat);

            zipStat->size = getStreamSize( is );
            zipStat->valid |= ZIP_STAT_SIZE;
            assert( !is.fail() );

            return sizeof(zip_stat_t);
        }

        case ZIP_SOURCE_FREE:
            return 0;

        default:
            ;
    }
    assert( false );
    return -1;
}

Expected<void> decompressZip_( zip_t * zip, const std::filesystem::path& targetFolder, const char * password )
{
    assert( zip );

    std::error_code ec;
    if ( !std::filesystem::is_directory( targetFolder, ec ) )
        return unexpected( "Directory does not exist " + utf8string( targetFolder ) );

    if ( password )
        zip_set_default_password( zip, password );

    zip_stat_t stats;
    zip_file_t* zfile;
    std::vector<char> fileBufer;
    for ( int i = 0; i < zip_get_num_entries( zip, 0 ); ++i )
    {
        if ( zip_stat_index( zip, i, 0, &stats ) == -1 )
            return unexpected( "Cannot process zip content" );

        std::string nameFixed = stats.name;
        std::replace( nameFixed.begin(), nameFixed.end(), '\\', '/' );
        std::filesystem::path relativeName = pathFromUtf8( nameFixed );
        relativeName.make_preferred();
        std::filesystem::path newItemPath = targetFolder / relativeName;
        if ( !nameFixed.empty() && nameFixed.back() == '/' )
        {
            if ( !std::filesystem::exists( newItemPath.parent_path(), ec ) )
                if ( !std::filesystem::create_directories( newItemPath.parent_path(), ec ) )
                    return unexpected( "Cannot create folder " + utf8string( newItemPath.parent_path() ) );
        }
        else
        {
            zfile = zip_fopen_index(zip,i,0);
            if ( !zfile )
                return unexpected( "Cannot open zip file " + nameFixed );

            // in some manually created zip-files there is no folder entries for files in sub-folders;
            // so let us create directory each time before saving a file in it
            if ( !std::filesystem::exists( newItemPath.parent_path(), ec ) )
                if ( !std::filesystem::create_directories( newItemPath.parent_path(), ec ) )
                    return unexpected( "Cannot create folder " + utf8string( newItemPath.parent_path() ) );

            std::ofstream ofs( newItemPath, std::ios::binary );
            if ( !ofs || ofs.bad() )
                return unexpected( "Cannot create file " + utf8string( newItemPath ) );

            fileBufer.resize(stats.size);
            auto bitesRead = zip_fread(zfile,(void*)fileBufer.data(),fileBufer.size());
            if ( bitesRead != (zip_int64_t)stats.size )
                return unexpected( "Cannot read file from zip " + nameFixed );

            zip_fclose(zfile);
            if ( !ofs.write( fileBufer.data(), fileBufer.size() ) )
                return unexpected( "Cannot write file from zip " + utf8string( newItemPath ) );
            ofs.close();
        }
    }
    return {};
}

} // anonymous namespace

Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder,
    const std::vector<std::filesystem::path>& excludeFiles, const char * password, ProgressCallback cb )
{
    MR_TIMER

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    std::error_code ec;
    if ( !std::filesystem::is_directory( sourceFolder, ec ) )
        return unexpected( "Directory '" + utf8string( sourceFolder ) + "' does not exist" );

    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err, subprogress( cb, 0.5f, 1.0f ) );    
    if ( !zip )
        return unexpected( "Cannot create zip, error code: " + std::to_string( err ) );

    auto goodFile = [&]( const std::filesystem::path & path )
    {
        if ( !is_regular_file( path, ec ) )
            return false;
        auto excluded = std::find_if( excludeFiles.begin(), excludeFiles.end(), [&] ( const auto& a )
        {
            return std::filesystem::equivalent( a, path, ec );
        } );
        return excluded == excludeFiles.end();
    };

    // pass #1: add directories in the archive and count the files
    int totalFiles = 0;
    for ( auto entry : DirectoryRecursive{ sourceFolder, ec } )
    {
        const auto path = entry.path();
        if ( entry.is_directory( ec ) && path != sourceFolder )
        {
            auto archiveDirPath = utf8string( std::filesystem::relative( path, sourceFolder, ec ) );
            // convert folder separators in Linux style for the latest 7-zip to open archive correctly
            std::replace( archiveDirPath.begin(), archiveDirPath.end(), '\\', '/' );
            if ( zip_dir_add( zip, archiveDirPath.c_str(), ZIP_FL_ENC_UTF_8 ) == -1 )
                return unexpected( "Cannot add directory " + archiveDirPath + " to archive" );
            continue;
        }

        if ( goodFile( path ) )
            ++totalFiles;
    }

    // pass #2: add files in the archive
    int compressedFiles = 0;
    auto scb = subprogress( cb, 0.0f, 0.5f );
    for ( auto entry : DirectoryRecursive{ sourceFolder, ec } )
    {
        const auto path = entry.path();
        if ( !goodFile( path ) )
            continue;

        auto fileSource = zip_source_file( zip, utf8string( path ).c_str(), 0, 0 );
        if ( !fileSource )
            return unexpected( "Cannot open file " + utf8string( path ) + " for reading" );

        auto archiveFilePath = utf8string( std::filesystem::relative( path, sourceFolder, ec ) );
        // convert folder separators in Linux style for the latest 7-zip to open archive correctly
        std::replace( archiveFilePath.begin(), archiveFilePath.end(), '\\', '/' );
        const auto index = zip_file_add( zip, archiveFilePath.c_str(), fileSource, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8 );
        if ( index < 0 )
        {
            zip_source_free( fileSource );
            return unexpected( "Cannot add file " + archiveFilePath + " to archive" );
        }

        if ( password )
        {
            if ( zip_file_set_encryption( zip, index, ZIP_EM_AES_256, password ) )
                return unexpected( "Cannot encrypt file " + archiveFilePath + " in archive" );
        }

        ++compressedFiles;
        if ( !reportProgress( scb, std::min( float( compressedFiles ) / totalFiles, 1.0f ) ) )
            return unexpectedOperationCanceled();
    }

    auto closeRes = zip.close();

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( closeRes == -1 )
        return unexpected( "Cannot close zip" );   

    return {};
}

Expected<void> decompressZip( const std::filesystem::path& zipFile, const std::filesystem::path& targetFolder, const char * password )
{
    MR_TIMER
    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_RDONLY, &err );
    if ( !zip )
        return unexpected( "Cannot open zip, error code: " + std::to_string( err ) );

    return decompressZip_( zip, targetFolder, password );
}

Expected<void> decompressZip( std::istream& zipStream, const std::filesystem::path& targetFolder, const char * password )
{
    MR_TIMER

    auto zipSource = zip_source_function_create( istreamZipSourceCallback, &zipStream, nullptr );
    if ( !zipSource )
        return unexpected( "Cannot create zip source from stream" );

    AutoCloseZip zip( *zipSource, ZIP_RDONLY, nullptr );
    if ( !zip )
        return unexpected( "Cannot open zip from source" );

    return decompressZip_( zip, targetFolder, password );
}

} // namespace MR
