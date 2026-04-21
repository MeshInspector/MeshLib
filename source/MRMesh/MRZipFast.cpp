#include "MRZip.h"
#include "MRDirectory.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#include <libdeflate.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

// Unencrypted compressZip fast path.
//
// compressZip() in MRZip.cpp delegates here whenever the caller didn't set a
// password. We write the ZIP container ourselves (local file headers, central
// directory, end-of-central-directory, plus ZIP64 records when needed) and
// hand the deflate primitive + CRC-32 computation to libdeflate. libdeflate is
// ~20-40% faster than zlib-ng-compat on the deflate hot path, with a better
// compression ratio at its highest levels, and does not depend on runtime SIMD
// feature detection -- so the speedup holds in Debug builds as well.
//
// Encrypted archives (settings.password non-empty) remain on libzip because
// WinZip AES requires PBKDF2-HMAC-SHA1 + AES-CTR + HMAC-SHA1 on top of the
// container, which libdeflate does not provide. Decompression also stays on
// libzip (ZIP reader compatibility matters more than speed there, and no one
// asked for faster decompression).

namespace MR
{

namespace
{

// ZIP format signatures and constants (APPNOTE.TXT).
constexpr uint32_t kLfhSig             = 0x04034b50u;
constexpr uint32_t kCdhSig             = 0x02014b50u;
constexpr uint32_t kEocdSig            = 0x06054b50u;
constexpr uint32_t kZip64EocdSig       = 0x06064b50u;
constexpr uint32_t kZip64EocdLocSig    = 0x07064b50u;
constexpr uint16_t kGpbfUtf8           = 0x0800u;   // bit 11: UTF-8 filenames
constexpr uint16_t kZip64ExtraId       = 0x0001u;
constexpr uint16_t kCompressionStore   = 0u;
constexpr uint16_t kCompressionDeflate = 8u;
constexpr uint16_t kVersionNeeded20    = 20u;
constexpr uint16_t kVersionNeeded45    = 45u;       // ZIP64
constexpr uint32_t kU32Max             = 0xFFFFFFFFu;
constexpr uint16_t kU16Max             = 0xFFFFu;

struct EntryMeta
{
    std::string  archiveName;       // UTF-8, forward slashes, trailing '/' for dirs
    uint64_t     offsetOfLFH   = 0; // offset of local file header from archive start
    uint64_t     uncompressed  = 0;
    uint64_t     compressed    = 0;
    uint32_t     crc32         = 0;
    uint16_t     method        = kCompressionStore;
    uint32_t     dosDateTime   = 0; // low16=time, high16=date
    bool         isDirectory   = false;
};

// A tiny write-only sink over std::ofstream that tracks byte offset.
struct SinkFile
{
    std::ofstream ofs;
    uint64_t      offset = 0;

    bool write( const void* data, size_t n )
    {
        ofs.write( static_cast<const char*>( data ), std::streamsize( n ) );
        offset += n;
        return bool( ofs );
    }
    bool u16( uint16_t v )
    {
        const uint8_t b[2] = { uint8_t( v ), uint8_t( v >> 8 ) };
        return write( b, 2 );
    }
    bool u32( uint32_t v )
    {
        const uint8_t b[4] = {
            uint8_t( v ), uint8_t( v >> 8 ), uint8_t( v >> 16 ), uint8_t( v >> 24 )
        };
        return write( b, 4 );
    }
    bool u64( uint64_t v )
    {
        uint8_t b[8];
        for ( int i = 0; i < 8; ++i )
            b[i] = uint8_t( v >> ( 8 * i ) );
        return write( b, 8 );
    }
};

// Convert filesystem mtime to DOS date/time (high16=date, low16=time).
// Falls back to 1980-01-01 00:00:00 when the clock is unavailable or out of range.
uint32_t toDosDateTime( const std::filesystem::path& p )
{
    const uint32_t kFallback = ( 1u << 16 ) | ( 1u << 21 ); // 1980-01-01 00:00:00

    std::error_code ec;
    auto ftime = std::filesystem::last_write_time( p, ec );
    if ( ec )
        return kFallback;

    // Convert file_time_type -> system_clock via the instantaneous offset.
    // DOS time has 2-second resolution so the tiny skew from this approach is
    // below the encodable precision; avoids the portability pitfalls of
    // std::chrono::file_clock::to_sys() on older clang/libc++ releases.
    const auto nowFile = std::filesystem::file_time_type::clock::now();
    const auto nowSys  = std::chrono::system_clock::now();
    const auto sys = nowSys + std::chrono::duration_cast<std::chrono::system_clock::duration>(
        ftime - nowFile );

    std::time_t tt = std::chrono::system_clock::to_time_t( sys );
    std::tm tm{};
#ifdef _WIN32
    if ( localtime_s( &tm, &tt ) != 0 )
        return kFallback;
#else
    if ( !localtime_r( &tt, &tm ) )
        return kFallback;
#endif

    int year = tm.tm_year + 1900;
    if ( year < 1980 || year > 2107 )
        return kFallback;

    const uint16_t date = uint16_t(
        ( ( year - 1980 ) << 9 ) | ( ( tm.tm_mon + 1 ) << 5 ) | tm.tm_mday );
    const uint16_t time = uint16_t(
        ( tm.tm_hour << 11 ) | ( tm.tm_min << 5 ) | ( tm.tm_sec / 2 ) );
    return ( uint32_t( date ) << 16 ) | uint32_t( time );
}

bool writeLocalFileHeader( SinkFile& s, const EntryMeta& e, bool zip64 )
{
    const uint16_t versionNeeded = zip64 ? kVersionNeeded45 : kVersionNeeded20;
    const uint32_t compSize32    = zip64 ? kU32Max : uint32_t( e.compressed );
    const uint32_t uncompSize32  = zip64 ? kU32Max : uint32_t( e.uncompressed );

    std::vector<uint8_t> extra;
    if ( zip64 )
    {
        // Mandatory pair: uncompressed then compressed. 16 bytes of payload.
        const uint16_t id  = kZip64ExtraId;
        const uint16_t len = 16;
        extra.resize( 4 + 16 );
        std::memcpy( extra.data() + 0, &id,  2 );
        std::memcpy( extra.data() + 2, &len, 2 );
        std::memcpy( extra.data() + 4,  &e.uncompressed, 8 );
        std::memcpy( extra.data() + 12, &e.compressed,   8 );
    }

    const uint16_t nameLen  = uint16_t( e.archiveName.size() );
    const uint16_t extraLen = uint16_t( extra.size() );

    if ( !s.u32( kLfhSig ) )                           return false;
    if ( !s.u16( versionNeeded ) )                     return false;
    if ( !s.u16( kGpbfUtf8 ) )                         return false;
    if ( !s.u16( e.method ) )                          return false;
    if ( !s.u16( uint16_t( e.dosDateTime & 0xFFFFu ) ) ) return false;
    if ( !s.u16( uint16_t( e.dosDateTime >> 16 ) ) )   return false;
    if ( !s.u32( e.crc32 ) )                           return false;
    if ( !s.u32( compSize32 ) )                        return false;
    if ( !s.u32( uncompSize32 ) )                      return false;
    if ( !s.u16( nameLen ) )                           return false;
    if ( !s.u16( extraLen ) )                          return false;
    if ( nameLen && !s.write( e.archiveName.data(), nameLen ) ) return false;
    if ( extraLen && !s.write( extra.data(), extra.size() ) )   return false;
    return true;
}

// Central-directory header: include a ZIP64 extra field whenever ANY of
// uncompressed/compressed/LFH-offset exceeds the 32-bit limit. We emit exactly
// the fields that are over-limit, in the spec-mandated order:
//   uncompressed (if >= 2^32), compressed (if >= 2^32), LFH offset (if >= 2^32).
bool writeCentralDirectoryHeader( SinkFile& s, const EntryMeta& e )
{
    const bool needU = e.uncompressed >= kU32Max;
    const bool needC = e.compressed   >= kU32Max;
    const bool needO = e.offsetOfLFH  >= kU32Max;
    const bool zip64 = needU || needC || needO;

    const uint16_t versionMadeBy = kVersionNeeded45;
    const uint16_t versionNeeded = zip64 ? kVersionNeeded45 : kVersionNeeded20;
    const uint32_t compSize32    = needC ? kU32Max : uint32_t( e.compressed );
    const uint32_t uncompSize32  = needU ? kU32Max : uint32_t( e.uncompressed );
    const uint32_t offset32      = needO ? kU32Max : uint32_t( e.offsetOfLFH );

    std::vector<uint8_t> extra;
    if ( zip64 )
    {
        std::vector<uint64_t> fields;
        if ( needU ) fields.push_back( e.uncompressed );
        if ( needC ) fields.push_back( e.compressed );
        if ( needO ) fields.push_back( e.offsetOfLFH );

        const uint16_t id  = kZip64ExtraId;
        const uint16_t len = uint16_t( fields.size() * 8 );
        extra.resize( 4 + len );
        std::memcpy( extra.data() + 0, &id,  2 );
        std::memcpy( extra.data() + 2, &len, 2 );
        for ( size_t i = 0; i < fields.size(); ++i )
            std::memcpy( extra.data() + 4 + i * 8, &fields[i], 8 );
    }

    const uint16_t nameLen  = uint16_t( e.archiveName.size() );
    const uint16_t extraLen = uint16_t( extra.size() );

    // External attrs: directory bit (0x10) in MS-DOS attrs; unix mode left at 0.
    const uint32_t externalAttrs = e.isDirectory ? 0x10u : 0u;

    if ( !s.u32( kCdhSig ) )                           return false;
    if ( !s.u16( versionMadeBy ) )                     return false;
    if ( !s.u16( versionNeeded ) )                     return false;
    if ( !s.u16( kGpbfUtf8 ) )                         return false;
    if ( !s.u16( e.method ) )                          return false;
    if ( !s.u16( uint16_t( e.dosDateTime & 0xFFFFu ) ) ) return false;
    if ( !s.u16( uint16_t( e.dosDateTime >> 16 ) ) )   return false;
    if ( !s.u32( e.crc32 ) )                           return false;
    if ( !s.u32( compSize32 ) )                        return false;
    if ( !s.u32( uncompSize32 ) )                      return false;
    if ( !s.u16( nameLen ) )                           return false;
    if ( !s.u16( extraLen ) )                          return false;
    if ( !s.u16( 0 ) )                                 return false; // comment len
    if ( !s.u16( 0 ) )                                 return false; // disk #
    if ( !s.u16( 0 ) )                                 return false; // internal attrs
    if ( !s.u32( externalAttrs ) )                     return false;
    if ( !s.u32( offset32 ) )                          return false;
    if ( nameLen && !s.write( e.archiveName.data(), nameLen ) ) return false;
    if ( extraLen && !s.write( extra.data(), extra.size() ) )   return false;
    return true;
}

// Whole-file deflate via libdeflate at the requested level. Returns the
// compressed payload; if deflate is worse than store, falls back to store
// (method is set accordingly on the entry).
struct DeflateResult
{
    std::vector<uint8_t> data;
    uint16_t             method = kCompressionStore;
};
DeflateResult deflateWholeFile( const std::vector<uint8_t>& raw, int level )
{
    DeflateResult out;
    if ( raw.empty() )
    {
        out.method = kCompressionStore;
        return out;
    }

    libdeflate_compressor* c = libdeflate_alloc_compressor( level );
    if ( !c )
    {
        out.data   = raw;
        out.method = kCompressionStore;
        return out;
    }

    const size_t bound = libdeflate_deflate_compress_bound( c, raw.size() );
    std::vector<uint8_t> buf( bound );
    const size_t produced = libdeflate_deflate_compress(
        c, raw.data(), raw.size(), buf.data(), buf.size() );
    libdeflate_free_compressor( c );

    if ( produced == 0 || produced >= raw.size() )
    {
        out.data   = raw;
        out.method = kCompressionStore;
    }
    else
    {
        buf.resize( produced );
        out.data   = std::move( buf );
        out.method = kCompressionDeflate;
    }
    return out;
}

} // namespace

// Public entry point for the libdeflate-backed ZIP writer. Returns an error
// when the archive can't be produced; the libzip fallback in MRZip.cpp is used
// only when settings.password is set, not as an error-recovery path.
Expected<void> compressZipFast( const std::filesystem::path& zipFile,
                                const std::filesystem::path& sourceFolder,
                                const CompressZipSettings& settings )
{
    MR_TIMER;
    assert( settings.password.empty() );

    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    std::error_code ec;
    if ( !std::filesystem::is_directory( sourceFolder, ec ) )
        return unexpected( "Directory '" + utf8string( sourceFolder ) + "' does not exist" );

    SinkFile sink;
    sink.ofs.open( zipFile, std::ios::binary | std::ios::trunc );
    if ( !sink.ofs )
        return unexpected( "Cannot create zip " + utf8string( zipFile ) );

    auto goodFile = [&]( const std::filesystem::path& path )
    {
        if ( !is_regular_file( path, ec ) )
            return false;
        auto it = std::find_if( settings.excludeFiles.begin(), settings.excludeFiles.end(),
            [&]( const auto& a ) { return std::filesystem::equivalent( a, path, ec ); } );
        return it == settings.excludeFiles.end();
    };

    // Pass #1: count files (for progress) and collect directory entries.
    int totalFiles = 0;
    std::vector<EntryMeta> directories;
    for ( auto entry : DirectoryRecursive{ sourceFolder, ec } )
    {
        const auto path = entry.path();
        if ( entry.is_directory( ec ) && path != sourceFolder )
        {
            EntryMeta d;
            d.archiveName = utf8string( std::filesystem::relative( path, sourceFolder, ec ) );
            std::replace( d.archiveName.begin(), d.archiveName.end(), '\\', '/' );
            if ( d.archiveName.empty() || d.archiveName.back() != '/' )
                d.archiveName += '/';
            d.isDirectory = true;
            d.method      = kCompressionStore;
            d.dosDateTime = toDosDateTime( path );
            directories.push_back( std::move( d ) );
            continue;
        }
        if ( goodFile( path ) )
            ++totalFiles;
    }

    const int rawLevel      = settings.compressionLevel;
    const int deflateLevel  = ( rawLevel <= 0 ) ? 6 : std::clamp( rawLevel, 1, 12 );

    // Pass #2: emit directory entries, then file entries.
    std::vector<EntryMeta> cdEntries;
    cdEntries.reserve( directories.size() + size_t( totalFiles ) );

    for ( auto& d : directories )
    {
        d.offsetOfLFH = sink.offset;
        const bool needZip64 = d.offsetOfLFH >= kU32Max;
        if ( !writeLocalFileHeader( sink, d, needZip64 ) )
            return unexpected( "Cannot write directory entry " + d.archiveName );
        cdEntries.push_back( d );
    }

    int compressed = 0;
    auto scb = subprogress( settings.cb, 0.0f, 0.95f );
    for ( auto entry : DirectoryRecursive{ sourceFolder, ec } )
    {
        const auto path = entry.path();
        if ( !goodFile( path ) )
            continue;

        // Read the whole file (libdeflate's non-streaming API expects an
        // in-memory buffer). Memory ceiling == largest single file in the
        // source tree; acceptable for typical MeshLib scenes.
        std::ifstream ifs( path, std::ios::binary | std::ios::ate );
        if ( !ifs )
            return unexpected( "Cannot open file " + utf8string( path ) + " for reading" );
        const std::streamoff size = ifs.tellg();
        if ( size < 0 )
            return unexpected( "Cannot size file " + utf8string( path ) );
        ifs.seekg( 0 );
        auto raw = std::vector<uint8_t>( size_t( size ) );
        if ( size > 0 && !ifs.read( reinterpret_cast<char*>( raw.data() ), size ) )
            return unexpected( "Cannot read file " + utf8string( path ) );

        EntryMeta e;
        e.archiveName = utf8string( std::filesystem::relative( path, sourceFolder, ec ) );
        std::replace( e.archiveName.begin(), e.archiveName.end(), '\\', '/' );
        e.uncompressed = raw.size();
        e.crc32        = libdeflate_crc32( 0, raw.data(), raw.size() );
        e.dosDateTime  = toDosDateTime( path );

        DeflateResult dr = deflateWholeFile( raw, deflateLevel );
        e.method       = dr.method;
        e.compressed   = dr.data.size();
        e.offsetOfLFH  = sink.offset;

        const bool needZip64 = e.uncompressed >= kU32Max
                            || e.compressed   >= kU32Max
                            || e.offsetOfLFH  >= kU32Max;

        if ( !writeLocalFileHeader( sink, e, needZip64 ) )
            return unexpected( "Cannot write header for " + e.archiveName );
        if ( e.compressed && !sink.write( dr.data.data(), dr.data.size() ) )
            return unexpected( "Cannot write body for " + e.archiveName );

        cdEntries.push_back( std::move( e ) );

        ++compressed;
        if ( !reportProgress( scb, std::min( float( compressed ) / totalFiles, 1.0f ) ) )
            return unexpectedOperationCanceled();
    }

    // Central directory.
    const uint64_t cdOffset = sink.offset;
    for ( const auto& e : cdEntries )
        if ( !writeCentralDirectoryHeader( sink, e ) )
            return unexpected( "Cannot write central directory entry " + e.archiveName );
    const uint64_t cdSize = sink.offset - cdOffset;
    const uint64_t cdCount = cdEntries.size();

    const bool zip64Footer = cdOffset >= kU32Max || cdSize >= kU32Max || cdCount >= kU16Max;

    if ( zip64Footer )
    {
        // ZIP64 end of central directory record (spec: fixed 44-byte prefix,
        // size-of-record field counts all bytes AFTER the 12-byte sig+size).
        const uint64_t zip64EocdOffset = sink.offset;
        if ( !sink.u32( kZip64EocdSig ) )                 return unexpected( "EOCD write failed" );
        if ( !sink.u64( uint64_t( 44u - 12u ) ) )         return unexpected( "EOCD write failed" );
        if ( !sink.u16( kVersionNeeded45 ) )              return unexpected( "EOCD write failed" );
        if ( !sink.u16( kVersionNeeded45 ) )              return unexpected( "EOCD write failed" );
        if ( !sink.u32( 0 ) )                             return unexpected( "EOCD write failed" );
        if ( !sink.u32( 0 ) )                             return unexpected( "EOCD write failed" );
        if ( !sink.u64( cdCount ) )                       return unexpected( "EOCD write failed" );
        if ( !sink.u64( cdCount ) )                       return unexpected( "EOCD write failed" );
        if ( !sink.u64( cdSize ) )                        return unexpected( "EOCD write failed" );
        if ( !sink.u64( cdOffset ) )                      return unexpected( "EOCD write failed" );

        // ZIP64 end of central directory locator (20 bytes).
        if ( !sink.u32( kZip64EocdLocSig ) )              return unexpected( "EOCD locator write failed" );
        if ( !sink.u32( 0 ) )                             return unexpected( "EOCD locator write failed" );
        if ( !sink.u64( zip64EocdOffset ) )               return unexpected( "EOCD locator write failed" );
        if ( !sink.u32( 1 ) )                             return unexpected( "EOCD locator write failed" );
    }

    // Regular EOCD (always emitted; readers look for the 0x06054b50 sig).
    const uint16_t entriesU16 = ( cdCount >= kU16Max ) ? kU16Max : uint16_t( cdCount );
    const uint32_t cdSize32   = ( cdSize   >= kU32Max ) ? kU32Max : uint32_t( cdSize );
    const uint32_t cdOffset32 = ( cdOffset >= kU32Max ) ? kU32Max : uint32_t( cdOffset );
    if ( !sink.u32( kEocdSig ) )   return unexpected( "EOCD write failed" );
    if ( !sink.u16( 0 ) )          return unexpected( "EOCD write failed" ); // disk
    if ( !sink.u16( 0 ) )          return unexpected( "EOCD write failed" ); // disk of CD
    if ( !sink.u16( entriesU16 ) ) return unexpected( "EOCD write failed" );
    if ( !sink.u16( entriesU16 ) ) return unexpected( "EOCD write failed" );
    if ( !sink.u32( cdSize32 ) )   return unexpected( "EOCD write failed" );
    if ( !sink.u32( cdOffset32 ) ) return unexpected( "EOCD write failed" );
    if ( !sink.u16( 0 ) )          return unexpected( "EOCD write failed" ); // comment len

    sink.ofs.close();
    if ( sink.ofs.fail() )
        return unexpected( "Cannot finalize zip " + utf8string( zipFile ) );

    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return {};
}

} // namespace MR
