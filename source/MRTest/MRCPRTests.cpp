#include "MRMesh/MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__)
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRGTest.h"
#include <curl/curl.h>
#include <cpr/cpr.h>

constexpr int MAX_RETRIES = 10;
constexpr std::chrono::seconds COOLDOWN_PERIOD { 10 };

namespace
{
void logCurlInfo()
{
    // Initialize libcurl
    curl_global_init( CURL_GLOBAL_DEFAULT );

    // Get version information
    const curl_version_info_data* data = curl_version_info( CURLVERSION_NOW );

    // Check if data is valid and print the version string
    if ( data )
    {
        spdlog::info( "libcurl Version: {}", data->version );
    }
    else
    {
        spdlog::error( "Failed to get libcurl version information." );
    }

    // Clean up libcurl
    curl_global_cleanup();
}
}

TEST( MRViewer, CPRTestGet )
{
    logCurlInfo();

    std::string baseUrl = "https://postman-echo.com/get";
    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    cpr::Parameters parameters;
    for ( const auto& [key, val] : params )
        parameters.Add( cpr::Parameter( key, val ) );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        cpr::Session session;
        session.SetVerbose( cpr::Verbose( true ) );
        session.SetDebugCallback( std::function{ []( cpr::DebugCallback::InfoType, std::string data, intptr_t )
        {
            spdlog::info( data );
        } } );
        //session.SetSslOptions( cpr::SslOptions{ .ssl_no_revoke = true } );
        session.SetUrl( cpr::Url{ baseUrl } );
        session.SetTimeout( cpr::Timeout{ 3000 } );
        session.SetParameters( parameters );
        const auto resp = session.Get();// cpr::Get( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, parameters );
        auto code = resp.status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( resp.error )
            spdlog::warn( "curl error {}: {}", (int32_t)resp.error.code, resp.error.message );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
        std::this_thread::sleep_for( COOLDOWN_PERIOD );
    }
}

TEST( MRViewer, CPRTestPost )
{
    logCurlInfo();

    std::string baseUrl = "https://postman-echo.com/post";
    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    std::vector<cpr::Pair> pairs;
    pairs.reserve( params.size() );
    for ( const auto& [key, val] : params )
        pairs.push_back( { key,val } );

    cpr::Payload payload( pairs.begin(), pairs.end() );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        cpr::Session session;
        session.SetVerbose( cpr::Verbose( true ) );
        session.SetDebugCallback( std::function{ []( cpr::DebugCallback::InfoType, std::string data, intptr_t )
        {
            spdlog::info( data );
        } } );
        //session.SetSslOptions( cpr::SslOptions{ .ssl_no_revoke = true } );
        session.SetUrl( cpr::Url{ baseUrl } );
        session.SetTimeout( cpr::Timeout{ 3000 } );
        session.SetPayload( payload );
        const auto resp = session.Post();// cpr::Post( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, payload );
        auto code = resp.status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( resp.error )
            spdlog::warn( "curl error {}: {}", (int32_t)resp.error.code, resp.error.message );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
        std::this_thread::sleep_for( COOLDOWN_PERIOD );
    }
}
#endif
