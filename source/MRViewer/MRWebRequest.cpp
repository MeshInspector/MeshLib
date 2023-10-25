#include "MRWebRequest.h"
#ifndef MRMESH_NO_CPR
#include "MRCommandLoop.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"
#ifndef __EMSCRIPTEN__
#include <cpr/cpr.h>
#include <thread>
#else
#include <mutex>
#endif

#ifdef __EMSCRIPTEN__

namespace
{
std::mutex sCallbackStoreMutex;
int sCallbackTS = 0;
std::unordered_map<int,MR::WebRequest::ResponseCallback> sCallbacksMap;
}

extern "C" 
{
EMSCRIPTEN_KEEPALIVE int emsCallResponseCallback( const char* response, bool async, int callbackTS )
{
    using namespace MR;
    std::string resStr = response;
    Json::Value resJson;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( reader->parse( resStr.data(), resStr.data() + resStr.size(), &resJson, &error ) )
    {
        if ( !async )
        {
            std::unique_lock lock( sCallbackStoreMutex );
            sCallbacksMap[callbackTS]( resJson );
            sCallbacksMap.erase( callbackTS );
        }
        else
        {
            CommandLoop::appendCommand( [resJson,callbackTS] ()
            {
                std::unique_lock lock( sCallbackStoreMutex );
                sCallbacksMap[callbackTS]( resJson );
                sCallbacksMap.erase( callbackTS );
            } );
        }
    }
    else
    {
        spdlog::info(error);
    }
    return 1;
}
}
#endif

namespace MR
{

void WebRequest::clear()
{
    method_ = Method::Get;
    timeout_ = 10000;
    params_ = {};
    headers_ = {};
    body_ = {};
}

void WebRequest::setMethod( Method method )
{
    method_ = method;
}

void WebRequest::setTimeout( int timeoutMs )
{
    timeout_ = timeoutMs;
}

void WebRequest::setParameters( std::unordered_map<std::string, std::string> parameters )
{
    params_ = std::move( parameters );
}

void WebRequest::setHeaders( std::unordered_map<std::string, std::string> headers )
{
    headers_ = std::move( headers );
}

void WebRequest::setBody( std::string body )
{
    body_ = std::move( body );
}

void WebRequest::send( std::string urlP, const std::string & logName, ResponseCallback callback, bool async /*= true */ )
{
#ifndef __EMSCRIPTEN__
    cpr::Timeout tm = cpr::Timeout{ timeout_ };
    cpr::Body body = cpr::Body( body_ );

    cpr::Parameters params;
    for ( const auto& [key, value] : params_ )
        params.Add( { key, value } );

    cpr::Header headers;
    for ( const auto& [key, value] : headers_ )
        headers[key] = value;

    auto sendLambda = [tm, body, params, headers, method = method_, url = urlP]()
    {
        cpr::Response response;
        if ( method == Method::Get )
            response = cpr::Get( cpr::Url( url ), headers, params, body, tm );
        else
            response = cpr::Post( cpr::Url( url ), headers, params, body, tm );
        return response;
    };
    clear();
    if ( !async )
    {
        spdlog::info( "WebRequest  {}", logName.c_str() );
        auto res = sendLambda();
        spdlog::info( "WebResponse {}: {}", logName.c_str(), int( res.status_code ) );
        Json::Value resJson;
        resJson["url"] = urlP;
        resJson["code"] = int( res.status_code );
        resJson["text"] = res.text;
        resJson["error"] = res.error.message;
        callback( resJson );
    }
    else
    {
        std::thread requestThread = std::thread( [sendLambda, callback, logName, url = urlP] ()
        {
            spdlog::info( "WebRequest  {}", logName.c_str() );
            auto res = sendLambda();
            spdlog::info( "WebResponse {}: {}", logName.c_str(), int( res.status_code ) );
            if ( !res.status_line.empty() )
                spdlog::info( "WebResponse {}: {}", logName.c_str(), res.status_line );
            if ( !res.reason.empty() )
                spdlog::info( "WebResponse {}: {}", logName.c_str(), res.reason );
            if ( res.error )
                spdlog::info( "WebResponse {}: {} {}", logName.c_str(), int( res.error.code ), res.error.message );
            Json::Value resJson;
            resJson["url"] = url;
            resJson["code"] = int( res.status_code );
            resJson["text"] = res.text;
            resJson["error"] = res.error.message;
            CommandLoop::appendCommand( [callback, resJson] ()
            {
                callback( resJson );
            }, CommandLoop::StartPosition::AfterPluginInit );
        } );
        requestThread.detach();
    }
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    MAIN_THREAD_EM_ASM( {
        web_req_clear();
        web_req_timeout = $0;
        web_req_body = UTF8ToString( $1 );
        web_req_method = $2;
        }, timeout_, body_.c_str(), int( method_ ) );
    for ( const auto& [key, value] : params_ )
        MAIN_THREAD_EM_ASM( web_req_add_param( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );
    
    for ( const auto& [key, value] : headers_ )
        MAIN_THREAD_EM_ASM( web_req_add_header( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );

    if ( !urlP.empty() && urlP.back() == '/' )
        urlP = urlP.substr( 0, int( urlP.size() ) - 1 );

    std::unique_lock lock( sCallbackStoreMutex );
    int callbackTS = sCallbackTS;
    sCallbacksMap[sCallbackTS] = callback;
    sCallbackTS++;
    lock.unlock();

    MAIN_THREAD_EM_ASM( web_req_send( UTF8ToString( $0 ), $1, $2 ), urlP.c_str(), async, callbackTS );
#pragma clang diagnostic pop
#endif
}

Expected<Json::Value, std::string> parseResponse( const Json::Value& response )
{
    if ( response["code"].asInt() == 0 )
        return unexpected( "Bad internet connection." );
    if ( response["error"].isString() )
    {
        auto error = response["error"].asString();
        if ( !error.empty() && error != "OK" )
            return unexpected( error );
    }
    if ( response["code"].asInt() == 403 )
        return unexpected( "Connection to " + response["url"].asString() + " is forbidden." );
    std::string text;
    if ( !response["text"].isString() )
        return unexpected( "Unknown error." );

    text = response["text"].asString();

    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( !reader->parse( text.data(), text.data() + text.size(), &root, &error ) )
        return unexpected( "Unknown error." );
    if ( root["message"].isString() )
        return unexpected( root["message"].asString() );
    return root;
}

}
#endif
