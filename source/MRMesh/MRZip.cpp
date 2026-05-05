#include "MRZip.h"
#include "MRDirectory.h"
#include "MRIOParsing.h"
#include "MRParallelFor.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRZlib.h"

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#endif

#include <zip.h>
#include <zipconf.h>

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>

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

// pre-deflated archive entry; owned by the AutoCloseZip that will write it out, so that libzip's
// zip_close (fired from ~AutoCloseZip) still sees live memory while it drains the source callbacks
struct DeflatedEntry
{
    std::vector<uint8_t> data;  ///< raw DEFLATE bytes (RFC 1951, no zlib wrapper)
    ZlibCompressStats stats;    ///< CRC-32 + uncompressed/compressed sizes from zlibCompressStream
    zip_uint16_t gpbFlags = 0;  ///< precomputed GPB bits 1-2 encoding the deflate level
    size_t pos = 0;             ///< read cursor for ZIP_SOURCE_READ
    zip_error_t err{};          ///< scratch for ZIP_SOURCE_ERROR
};

// PKZIP APPNOTE 4.4.4 buckets the deflate level into four coarse categories via GPB bits 1 and 2;
// levels 6-8 all map to "normal" and levels 2-5 all map to "fast" — the spec has no finer resolution
zip_uint16_t deflateGpbLevelFlags( int level )
{
    if ( level == 9 ) return 0b0010;                   // maximum:   bit 1 set
    if ( level >= 2 && level <= 5 ) return 0b0100;     // fast:      bit 2 set
    if ( level == 1 ) return 0b0110;                   // superfast: bits 1 + 2 set
    return 0b0000;                                     // normal (levels 6-8): neither bit set
}

// zip_source_function callback for a pre-deflated buffer; lets libzip's trust-source
// fast path in zip_close() copy the bytes into the archive without any recompression
// (no SEEK/TELL on purpose — keeps libzip on the linear-read path)
zip_int64_t deflatedSourceCallback( void* user, void* data, zip_uint64_t len, zip_source_cmd_t cmd )
{
    if ( !user )
    {
        assert( false );
        return -1;
    }
    auto* entry = static_cast<DeflatedEntry*>( user );
    switch ( cmd )
    {
        case ZIP_SOURCE_SUPPORTS:
            return zip_source_make_command_bitmap( ZIP_SOURCE_OPEN, ZIP_SOURCE_READ, ZIP_SOURCE_CLOSE,
                ZIP_SOURCE_STAT, ZIP_SOURCE_ERROR, ZIP_SOURCE_FREE, ZIP_SOURCE_SUPPORTS,
                ZIP_SOURCE_GET_FILE_ATTRIBUTES, -1 );

        case ZIP_SOURCE_OPEN:
            entry->pos = 0;
            return 0;

        case ZIP_SOURCE_READ:
        {
            const size_t n = std::min<size_t>( size_t( len ), entry->data.size() - entry->pos );
            if ( n )
                std::memcpy( data, entry->data.data() + entry->pos, n );
            entry->pos += n;
            return zip_int64_t( n );
        }

        case ZIP_SOURCE_CLOSE:
            return 0;

        case ZIP_SOURCE_STAT:
        {
            auto* st = (zip_stat_t*)data;
            zip_stat_init( st );
            st->size = entry->stats.uncompressedSize;
            st->comp_size = entry->stats.compressedSize;
            st->comp_method = ZIP_CM_DEFLATE;
            st->crc = entry->stats.crc32;
            st->encryption_method = ZIP_EM_NONE;
            st->valid = ZIP_STAT_SIZE | ZIP_STAT_COMP_SIZE
                      | ZIP_STAT_COMP_METHOD | ZIP_STAT_CRC
                      | ZIP_STAT_ENCRYPTION_METHOD;
            return sizeof( zip_stat_t );
        }

        case ZIP_SOURCE_ERROR:
            return zip_error_to_data( &entry->err, data, len );

        case ZIP_SOURCE_FREE:
            return 0; // entry is owned by the AutoCloseZip, not by libzip

        case ZIP_SOURCE_GET_FILE_ATTRIBUTES:
        {
            auto* a = (zip_file_attributes_t*)data;
            zip_file_attributes_init( a );
            a->valid = ZIP_FILE_ATTRIBUTES_VERSION_NEEDED
                     | ZIP_FILE_ATTRIBUTES_GENERAL_PURPOSE_BIT_FLAGS;
            a->version_needed = 20;
            a->general_purpose_bit_flags = entry->gpbFlags;
            a->general_purpose_bit_mask = 0b0110; // only bits 1-2 (deflate level) are authoritative here
            return sizeof( zip_file_attributes_t );
        }

        default:
            return -1;
    }
}

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

    /// Takes ownership of pre-deflated entries and registers them in the archive in order.
    /// Entries stay alive for the AutoCloseZip's lifetime, so libzip's later zip_close (either
    /// explicit or from the destructor) still sees valid memory when it drains the sources.
    Expected<void> addPreDeflatedEntries(
        std::vector<DeflatedEntry> entries,
        const std::vector<std::pair<std::filesystem::path, std::string>>& files,
        int level,
        const std::string& password )
    {
        assert( entries.size() == files.size() );
        entries_ = std::move( entries );
        const zip_uint16_t gpbFlags = deflateGpbLevelFlags( level );
        for ( size_t i = 0; i < entries_.size(); ++i )
        {
            auto& entry = entries_[i];
            entry.gpbFlags = gpbFlags;
            const auto& archiveFilePath = files[i].second;

            zip_source_t* src = zip_source_function( handle_, deflatedSourceCallback, &entry );
            if ( !src )
                return unexpected( "Cannot create source for " + archiveFilePath );

            const auto index = zip_file_add( handle_, archiveFilePath.c_str(), src, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8 );
            if ( index < 0 )
            {
                zip_source_free( src );
                return unexpected( "Cannot add file " + archiveFilePath + " to archive" );
            }

            zip_set_file_compression( handle_, index, ZIP_CM_DEFLATE, zip_uint32_t( level ) );

            if ( !password.empty() )
            {
                if ( zip_file_set_encryption( handle_, index, ZIP_EM_AES_256, password.c_str() ) )
                    return unexpected( "Cannot encrypt file " + archiveFilePath + " in archive" );
            }
        }
        return {};
    }

private:
    zip_t * handle_ = nullptr;
    ProgressData pd_;
    std::vector<DeflatedEntry> entries_;
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

Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder, const CompressZipSettings& settings )
{
    MR_TIMER;

    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    std::error_code ec;
    if ( !std::filesystem::is_directory( sourceFolder, ec ) )
        return unexpected( "Directory '" + utf8string( sourceFolder ) + "' does not exist" );

    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err, subprogress( settings.cb, 0.8f, 1.0f ) );
    if ( !zip )
        return unexpected( "Cannot create zip, error code: " + std::to_string( err ) );

    auto goodFile = [&]( const std::filesystem::path & path )
    {
        if ( !is_regular_file( path, ec ) )
            return false;
        auto excluded = std::find_if( settings.excludeFiles.begin(), settings.excludeFiles.end(), [&] ( const auto& a )
        {
            return std::filesystem::equivalent( a, path, ec );
        } );
        return excluded == settings.excludeFiles.end();
    };

    // pass #1: add directories in the archive and collect the files to compress
    std::vector<std::pair<std::filesystem::path, std::string>> files;
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
        {
            auto archiveFilePath = utf8string( std::filesystem::relative( path, sourceFolder, ec ) );
            // convert folder separators in Linux style for the latest 7-zip to open archive correctly
            std::replace( archiveFilePath.begin(), archiveFilePath.end(), '\\', '/' );
            files.emplace_back( path, std::move( archiveFilePath ) );
        }
    }

    // pass #2: deflate each file in parallel, then hand libzip pre-deflated bytes via a source
    // callback — libzip's trust-source fast path copies them into the archive without recompressing.
    // level 0 (the settings-level "use default") normalizes to 6 — zlib's internal default is 6,
    // and APPNOTE has no sentinel for "default", so 6 is what the archive entry records
    const int level = settings.compressionLevel == 0 ? 6 : std::clamp( settings.compressionLevel, 1, 9 );

    // Phase A — parallel: read each file and deflate it into entries[i]. The first worker
    // to fail wins the hadError CAS and publishes its message; others see hadError and skip
    std::vector<DeflatedEntry> entries( files.size() );
    std::atomic<bool> hadError{ false };
    std::string firstError;
    auto scbA = subprogress( settings.cb, 0.0f, 0.8f );

    auto reportError = [&]( std::string msg )
    {
        bool expected = false;
        if ( hadError.compare_exchange_strong( expected, true, std::memory_order_acq_rel ) )
            firstError = std::move( msg );
    };

    auto keepGoing = ParallelFor( files, [&]( size_t i )
    {
        if ( hadError.load( std::memory_order_relaxed ) )
            return; // some other worker already failed; skip

        std::ifstream in( files[i].first, std::ios::binary );
        if ( !in )
            return reportError( "Cannot open file " + utf8string( files[i].first ) + " for reading" );

        auto& e = entries[i];
        std::ostringstream out( std::ios::binary );
        if ( auto r = zlibCompressStream( in, out,
                 ZlibCompressParams{ { .rawDeflate = true }, level, &e.stats } ); !r )
            return reportError( r.error() );
        // zlibCompressStream requires std::ostream&, so we go through ostringstream and copy out;
        // the copy is O(deflated_size), trivial next to zlib's own cost per file
        const std::string s = std::move( out ).str();
        e.data.assign( s.begin(), s.end() );
    }, scbA, 1 /* reportProgressEvery — tick per file for UX and cancel responsiveness */ );

    if ( !keepGoing )
        return unexpectedOperationCanceled();
    if ( hadError.load() )
        return unexpected( std::move( firstError ) );

    // Phase B — serial hand-off to libzip; the AutoCloseZip keeps entries alive through zip_close
    if ( auto res = zip.addPreDeflatedEntries( std::move( entries ), files, level, settings.password ); !res )
        return res;

    auto closeRes = zip.close();

    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( closeRes == -1 )
        return unexpected( "Cannot close zip" );

    return {};
}

Expected<void> compressZip( const std::filesystem::path& zipFile, const std::filesystem::path& sourceFolder,
    const std::vector<std::filesystem::path>& excludeFiles, const char* password, ProgressCallback cb )
{
    return compressZip( zipFile, sourceFolder, { .excludeFiles = excludeFiles ,.password = std::string( password ),.cb = cb } );
}

Expected<void> decompressZip( const std::filesystem::path& zipFile, const std::filesystem::path& targetFolder, const char * password )
{
    MR_TIMER;
    int err;
    AutoCloseZip zip( utf8string( zipFile ).c_str(), ZIP_RDONLY, &err );
    if ( !zip )
        return unexpected( "Cannot open zip, error code: " + std::to_string( err ) );

    return decompressZip_( zip, targetFolder, password );
}

Expected<void> decompressZip( std::istream& zipStream, const std::filesystem::path& targetFolder, const char * password )
{
    MR_TIMER;

    auto zipSource = zip_source_function_create( istreamZipSourceCallback, &zipStream, nullptr );
    if ( !zipSource )
        return unexpected( "Cannot create zip source from stream" );

    AutoCloseZip zip( *zipSource, ZIP_RDONLY, nullptr );
    if ( !zip )
        return unexpected( "Cannot open zip from source" );

    return decompressZip_( zip, targetFolder, password );
}

} // namespace MR
