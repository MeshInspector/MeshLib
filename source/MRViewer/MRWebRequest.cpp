#include "MRWebRequest.h"
#include "MRCommandLoop.h"
#include <cpr/cpr.h>
#include <thread>

namespace MR
{

bool WebRequest::readyForNextRequest()
{
    return instance_().requestReady_;
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
    if ( !inst.requestReady_ )
    {
        assert( false );
        return false;
    }

    inst.requestReady_ = false;

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
        inst.requestReady_ = true;
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
            CommandLoop::appendCommand( [callback, resJson, &inst] ()
            {
                callback( resJson );
                inst.requestReady_ = true;
            } );
        } );
        requestThread.detach();
    }
    return true;
}

MR::WebRequest& WebRequest::instance_()
{
    static WebRequest inst;
    return inst;
}

}