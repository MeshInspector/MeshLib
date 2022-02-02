#ifndef __EMSCRIPTEN__
#include <cpr/cpr.h>
#include <spdlog/spdlog.h>
#include "MRGTest.h"

constexpr int MAX_RETRIES = 10;

TEST( CPRTest, Get )
{
    std::string baseUrl = "https://postman-echo.com/get";
    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    cpr::Parameters parameters;
    for ( const auto& [key, val] : params )
        parameters.Add( cpr::Parameter( key, val ) );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        auto code = cpr::Get( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, parameters ).status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
    }
}

TEST( CPRTest, Post )
{
    std::string baseUrl = "https://postman-echo.com/post";
    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    std::vector<cpr::Pair> pairs;
    pairs.reserve( params.size() );
    for ( const auto& [key, val] : params )
        pairs.push_back( { key,val } );

    cpr::Payload payload( pairs.begin(), pairs.end() );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        auto code = cpr::Post( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, payload ).status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
    }
}
#endif