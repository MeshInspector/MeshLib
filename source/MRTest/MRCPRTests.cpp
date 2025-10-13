#include "MRMesh/MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__)
#include <cpr/cpr.h>
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRGTest.h"

constexpr int MAX_RETRIES = 10;
constexpr std::chrono::seconds COOLDOWN_PERIOD { 10 };

TEST( MRViewer, CPRTestGet )
{
    std::string baseUrl = "https://postman-echo.com/get";
    if ( const char * overrideUrl = std::getenv( "MRTEST_OVERRIDE_ECHO_SERVER_URL" ) )
        baseUrl = overrideUrl;

    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    cpr::Parameters parameters;
    for ( const auto& [key, val] : params )
        parameters.Add( cpr::Parameter( key, val ) );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        const auto resp = cpr::Get( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, parameters );
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
    std::string baseUrl = "https://postman-echo.com/post";
    if ( const char * overrideUrl = std::getenv( "MRTEST_OVERRIDE_ECHO_SERVER_URL" ) )
        baseUrl = overrideUrl;

    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    std::vector<cpr::Pair> pairs;
    pairs.reserve( params.size() );
    for ( const auto& [key, val] : params )
        pairs.push_back( { key,val } );

    cpr::Payload payload( pairs.begin(), pairs.end() );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        const auto resp = cpr::Post( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, payload );
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
