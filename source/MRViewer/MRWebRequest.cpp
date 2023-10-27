#include "MRWebRequest.h"
#ifndef MRMESH_NO_CPR
#include "MRCommandLoop.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"

#ifndef __EMSCRIPTEN__
#include <cpr/cpr.h>
#include <fstream>
#include <optional>
#include <thread>
#else
#include <mutex>
#endif

namespace
{

struct RequestContext
{
#ifdef __EMSCRIPTEN__
    MR::WebRequest::ResponseCallback callback;
#else
    std::optional<std::ofstream> output;
#endif
};

std::mutex sRequestContextMutex;
int sRequestContextCounter = 0;
std::unordered_map<int, std::unique_ptr<RequestContext>> sRequestContextMap;

}

#ifdef __EMSCRIPTEN__
extern "C"
{
EMSCRIPTEN_KEEPALIVE int emsCallResponseCallback( const char* response, bool async, int ctxId )
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
            auto& ctx = sRequestContextMap.at( ctxId );
            ctx->callback( resJson );

            std::unique_lock lock( sRequestContextMutex );
            sRequestContextMap.erase( ctxId );
        }
        else
        {
            CommandLoop::appendCommand( [resJson, ctxId] ()
            {
                auto& ctx = sRequestContextMap.at( ctxId );
                ctx->callback( resJson );

                std::unique_lock lock( sRequestContextMutex );
                sRequestContextMap.erase( ctxId );
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
#else
namespace
{

bool downloadFileCallback( std::string data, intptr_t userdata )
{
    const auto ctxId = (int)userdata;
    auto& ctx = sRequestContextMap.at( ctxId );
    assert( ctx->output.has_value() );
    *ctx->output << data;
    return true;
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
    formData_ = {};
    body_ = {};
    outputPath_ = {};
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

void WebRequest::setFormData( std::vector<FormData> formData )
{
    formData_ = std::move( formData );
}

void WebRequest::setBody( std::string body )
{
    body_ = std::move( body );
}

void WebRequest::setOutputPath( std::string outputPath )
{
    outputPath_ = std::move( outputPath );
}

void WebRequest::send( std::string urlP, const std::string & logName, ResponseCallback callback, bool async /*= true */ )
{
    int ctxId;
    RequestContext* ctx;
    {
        std::unique_lock lock( sRequestContextMutex );
        ctxId = sRequestContextCounter++;
        auto [it, _] = sRequestContextMap.emplace( ctxId, std::make_unique<RequestContext>() );
        ctx = it->second.get();
    }

#ifndef __EMSCRIPTEN__
    cpr::Timeout tm = cpr::Timeout{ timeout_ };
    cpr::Body body = cpr::Body( body_ );

    cpr::Parameters params;
    for ( const auto& [key, value] : params_ )
        params.Add( { key, value } );

    cpr::Header headers;
    for ( const auto& [key, value] : headers_ )
        headers[key] = value;

    cpr::Multipart multipart( {} );
    for ( const auto& formData : formData_ )
        // TODO: update libcpr to support custom file names
        multipart.parts.emplace_back( formData.name, cpr::File( formData.path ), formData.contentType );

    if ( !outputPath_.empty() )
    {
        ctx->output = std::ofstream( outputPath_, std::ios::binary );
        if ( !ctx->output->good() )
        {
            spdlog::info( "WebResponse {}: Unable to open output file", logName.c_str() );
            return;
        }
    }

    auto sendLambda = [ctxId, ctx, tm, body, params, headers, multipart, method = method_, url = urlP] () -> cpr::Response
    {
        cpr::Session session;
        session.SetUrl( url );
        session.SetHeader( headers );
        session.SetParameters( params );
        session.SetTimeout( tm );
        if ( multipart.parts.empty() )
            session.SetBody( body );
        else
            session.SetMultipart( multipart );
        if ( ctx->output.has_value() )
            session.SetWriteCallback( { &downloadFileCallback, ctxId } );

        switch ( method )
        {
            case Method::Get:
                return session.Get();
            case Method::Post:
                return session.Post();
        }

#ifdef __cpp_lib_unreachable
        std::unreachable();
#else
        assert( false );
        return {};
#endif
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
    },
        timeout_,
        body_.c_str(),
        int( method_ )
    );

    for ( const auto& [key, value] : params_ )
        MAIN_THREAD_EM_ASM( web_req_add_param( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );
    
    for ( const auto& [key, value] : headers_ )
        MAIN_THREAD_EM_ASM( web_req_add_header( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );

    for ( const auto& formData : formData_ )
    {
        MAIN_THREAD_EM_ASM(
            web_req_add_formdata( UTF8ToString( $0 ), UTF8ToString( $1 ), UTF8ToString( $2 ), UTF8ToString( $3 ) ),
            formData.path.c_str(),
            formData.contentType.c_str(),
            formData.name.c_str(),
            formData.fileName.c_str()
        );
    }

    if ( !urlP.empty() && urlP.back() == '/' )
        urlP = urlP.substr( 0, int( urlP.size() ) - 1 );

    ctx->callback = callback;

    if ( outputPath_.empty() )
        MAIN_THREAD_EM_ASM( web_req_send( UTF8ToString( $0 ), $1, $2 ), urlP.c_str(), async, ctxId );
    else
        MAIN_THREAD_EM_ASM( web_req_async_download( UTF8ToString( $0 ), UTF8ToString( $1 ), $2 ), urlP.c_str(), outputPath_.c_str(), ctxId );
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
