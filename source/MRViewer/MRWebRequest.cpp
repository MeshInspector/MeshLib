#ifndef MRMESH_NO_CPR
#include "MRWebRequest.h"
#include "MRCommandLoop.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"
#ifndef __EMSCRIPTEN__
#include <cpr/cpr.h>
#include <thread>
#endif

namespace
{
bool sRequestReady{true};
}

#ifdef __EMSCRIPTEN__

namespace
{
MR::WebRequest::ResponseCallback sCallback;
bool sAsyncCall{false};
}

extern "C" 
{
EMSCRIPTEN_KEEPALIVE int emsCallResponseCallback( const char* response )
{
    using namespace MR;
    std::string resStr = response;
    Json::Value resJson;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
    std::string error;
    if ( reader->parse( resStr.data(), resStr.data() + resStr.size(), &resJson, &error ) )
    {
        if ( !sAsyncCall )
        {
            sCallback( resJson );
            sRequestReady = true;
        }
        else
        {
            CommandLoop::appendCommand( [resJson] ()
            {
                sCallback( resJson );
                sRequestReady = true;
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

bool WebRequest::readyForNextRequest()
{
    return sRequestReady;
}

void WebRequest::clear()
{
    auto& inst = instance_();
    inst.method_ = Method::Get;
    inst.timeout_ = 10000;
    inst.params_ = {};
    inst.headers_ = {};
    inst.body_ = {};
}

void WebRequest::setMethod( Method method )
{
    instance_().method_ = method;
}

void WebRequest::setTimeout( int timeoutMs )
{
    instance_().timeout_ = timeoutMs;
}

void WebRequest::setParameters( std::unordered_map<std::string, std::string> parameters )
{
    instance_().params_ = std::move( parameters );
}

void WebRequest::setHeaders( std::unordered_map<std::string, std::string> headers )
{
    instance_().headers_ = std::move( headers );
}

void WebRequest::setBody( std::string body )
{
    instance_().body_ = std::move( body );
}

bool WebRequest::send( std::string urlP, ResponseCallback callback, bool async /*= true */ )
{
    auto& inst = instance_();
    if ( !sRequestReady )
    {
        assert( false );
        return false;
    }

    sRequestReady = false;
#ifndef __EMSCRIPTEN__
    cpr::Timeout tm = cpr::Timeout{ inst.timeout_ };
    cpr::Body body = cpr::Body( inst.body_ );

    cpr::Parameters params;
    for ( const auto& [key, value] : inst.params_ )
        params.Add( { key, value } );

    cpr::Header headers;
    for ( const auto& [key, value] : inst.headers_ )
        headers[key] = value;

    auto sendLambda = [tm, body, params, headers, method = inst.method_, url = std::move( urlP )]()
    {
        cpr::Response response;
        if ( method == Method::Get )
            response = cpr::Get( cpr::Url( url ), headers, params, body, tm );
        else
            response = cpr::Post( cpr::Url( url ), headers, params, body, tm );
        return response;
    };
    inst.clear();
    if ( !async )
    {
        auto res = sendLambda();
        Json::Value resJson;
        resJson["code"] = int( res.status_code );
        resJson["text"] = res.text;
        resJson["error"] = res.error.message;
        callback( resJson );
        sRequestReady = true;
    }
    else
    {
        std::thread requestThread = std::thread( [sendLambda, callback, &inst] ()
        {
            auto res = sendLambda();
            Json::Value resJson;
            resJson["code"] = int( res.status_code );
            resJson["text"] = res.text;
            resJson["error"] = res.error.message;
            CommandLoop::appendCommand( [callback, resJson] ()
            {
                callback( resJson );
                sRequestReady = true;
            } );
        } );
        requestThread.detach();
    }
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( {
        web_req_clear();
        web_req_timeout = $0;
        web_req_body = UTF8ToString( $1 );
        web_req_method = $2;
        }, inst.timeout_, inst.body_.c_str(), int( inst.method_ ) );
    for ( const auto& [key, value] : inst.params_ )
        EM_ASM( web_req_add_param( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );
    
    for ( const auto& [key, value] : inst.headers_ )
        EM_ASM( web_req_add_header( UTF8ToString( $0 ), UTF8ToString( $1 ) ), key.c_str(), value.c_str() );
    
    sCallback = callback;
    sAsyncCall = async;

    EM_ASM( web_req_send( UTF8ToString( $0 ), $1 ), urlP.c_str(), async );
#pragma clang diagnostic pop
#endif
    return true;
}

MR::WebRequest& WebRequest::instance_()
{
    static WebRequest inst;
    return inst;
}

}
#endif
