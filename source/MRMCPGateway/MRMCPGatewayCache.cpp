#include "MRMCPGatewayCache.h"
#include "MRMCPGatewayBackend.h"
#include "MRMCPGatewayConfig.h"
#include "MRMCPGatewaySpawn.h"
#include "MRMCPGatewayUtf8.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <system_error>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

namespace MR::McpGateway
{

namespace
{

// Compile-time build stamp baked into this translation unit. Written to the
// cache file's `stamp` field after each successful prime; compared on startup
// to decide whether the existing cache is still good. To force a re-prime,
// touch this file (or just delete the cache dir).
constexpr const char* kBuildStamp = __TIMESTAMP__;

// PASTED from MR::getUserConfigDir() in MeshLib/source/MRMesh/MRSystem.cpp.
// Adapted to: hardcode the leaf folder (no Config::instance() dep) and use
// std::cerr instead of spdlog so the gateway keeps its zero-MRMesh,
// zero-spdlog dependency footprint.
std::filesystem::path gatewayUserConfigDir()
{
#if defined( _WIN32 )
    std::filesystem::path filepath( _wgetenv( L"APPDATA" ) );
#else
#if defined( __EMSCRIPTEN__ )
    std::filesystem::path filepath( "/" );
#else
    std::filesystem::path filepath;
    if ( const auto* pw = getpwuid( getuid() ) )
        filepath = pw->pw_dir;
    else if ( const char* h = std::getenv( "HOME" ) )
        filepath = h;
#endif
    filepath /= ".local";
    filepath /= "share";
#endif
    filepath /= "MRMCPGateway";
    std::error_code ec;
    if ( !std::filesystem::is_directory( filepath, ec ) || ec )
    {
        std::filesystem::create_directories( filepath, ec );
        if ( ec )
            std::cerr << "MRMCPGateway: cannot create " << pathToUtf8( filepath )
                      << ": " << ec.message() << "\n";
    }
    return filepath;
}

std::filesystem::path cacheDir( const Config& cfg )
{
    auto d = gatewayUserConfigDir();
    if ( !cfg.toolsCacheNamespace.empty() )
        d /= pathFromUtf8( cfg.toolsCacheNamespace );
    return d;
}

std::filesystem::path cachePath( const Config& cfg ) { return cacheDir( cfg ) / "mcp_tools_cache.json"; }

// Reads the cache file and returns true iff its `stamp` field equals kBuildStamp.
bool cacheStampMatches( const std::filesystem::path& cache )
{
    std::ifstream f( cache );
    if ( !f )
        return false;
    nlohmann::json doc;
    try { f >> doc; } catch ( ... ) { return false; }
    if ( !doc.is_object() || !doc.contains( "stamp" ) || !doc["stamp"].is_string() )
        return false;
    return doc["stamp"].get<std::string>() == kBuildStamp;
}

// Reads the cache, sets its `stamp` field to kBuildStamp, atomically writes it back.
bool embedStampInCache( const std::filesystem::path& cache )
{
    std::ifstream in( cache );
    if ( !in )
        return false;
    nlohmann::json doc;
    try { in >> doc; } catch ( ... ) { return false; }
    in.close();
    if ( !doc.is_object() )
        return false;
    doc["stamp"] = kBuildStamp;

    auto tmp = cache;
    tmp += ".tmp";
    {
        std::ofstream out( tmp );
        if ( !out )
            return false;
        out << doc.dump( 2 );
    }
    std::error_code ec;
    std::filesystem::rename( tmp, cache, ec );
    if ( ec )
    {
        std::filesystem::remove( tmp );
        return false;
    }
    return true;
}

} // anonymous namespace

std::vector<nlohmann::json>& getCachedTools()
{
    static std::vector<nlohmann::json> instance;
    return instance;
}

void ensureFreshCache( const Config& cfg )
{
    const auto cache = cachePath( cfg );

    if ( cacheStampMatches( cache ) )
        return; // already fresh

    // Backend already running → live tools/list will be authoritative this
    // session, skip the prime (and avoid port collision).
    if ( probeBackendAlive( cfg.targetUrl ) )
        return;

    if ( cfg.launchCommand.empty() )
        return;

    std::error_code ec;
    const auto mtimeBefore = std::filesystem::exists( cache, ec )
        ? std::filesystem::last_write_time( cache, ec )
        : std::filesystem::file_time_type{};

    std::vector<std::string> primeArgs = cfg.launchArgs;
    for ( const char* flag : { "-hidden", "-noEventLoop", "-noTelemetry", "-noSplash" } )
        primeArgs.emplace_back( flag );
    primeArgs.emplace_back( "-mcpPort" );
    primeArgs.emplace_back( std::to_string( cfg.mcpPort ) );
    primeArgs.emplace_back( "-mcpDumpFile" );
    // pathToUtf8 instead of cache.string(): Spawn pipes args through utf8ToWide,
    // and `path::string()` on Windows narrows via the system codepage, so
    // non-ASCII path components were getting mangled before reaching the backend.
    primeArgs.emplace_back( pathToUtf8( cache ) );

    // Synchronous spawn: blocks until the backend exits (or times out). Avoids
    // the polling-sees-stale-cache race that file-existence polling has when an
    // older cache file is already on disk.
    if ( !spawnAndWait( cfg.launchCommand, primeArgs, cfg.launchTimeout ) )
    {
        std::cerr << "MRMCPGateway: prime spawn did not complete cleanly\n";
        return;
    }

    // Confirm the backend actually rewrote the cache file before stamping it.
    if ( !std::filesystem::exists( cache, ec ) )
    {
        std::cerr << "MRMCPGateway: backend exited without writing cache file\n";
        return;
    }
    const auto mtimeAfter = std::filesystem::last_write_time( cache, ec );
    if ( mtimeAfter <= mtimeBefore )
    {
        std::cerr << "MRMCPGateway: cache file unchanged after prime; not stamping\n";
        return;
    }
    embedStampInCache( cache );
}

void loadCachedTools( const Config& cfg )
{
    auto& cachedTools = getCachedTools();
    cachedTools.clear();
    const auto cache = cachePath( cfg );
    std::ifstream f( cache );
    if ( !f )
        return;
    nlohmann::json doc;
    try { f >> doc; } catch ( ... ) { return; }
    if ( !doc.is_object() || !doc.contains( "tools" ) || !doc["tools"].is_array() )
        return;
    for ( const auto& t : doc["tools"] )
        if ( t.is_object() )
            cachedTools.push_back( t );
}

} // namespace MR::McpGateway
