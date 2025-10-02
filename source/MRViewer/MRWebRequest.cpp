#include "MRWebRequest.h"
#if defined( __EMSCRIPTEN__ ) || !defined( MRMESH_NO_CPR )
#include "MRCommandLoop.h"

#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRSerializer.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"

#ifndef __EMSCRIPTEN__
#include <cpr/cpr.h>
#include <fstream>
#include <optional>
#else
#include <mutex>
#endif

namespace
{

struct RequestContext
{
    MR::ProgressCallback uploadCallback;
    MR::ProgressCallback downloadCallback;
#ifdef __EMSCRIPTEN__
    MR::WebRequest::ResponseCallback responseCallback;
#else
    std::optional<std::ifstream> input;
    std::optional<std::ofstream> output;
#endif
};

std::mutex sRequestContextMutex;
int sRequestContextCounter = 0;
std::unordered_map<int, std::shared_ptr<RequestContext>> sRequestContextMap;

#ifdef __EMSCRIPTEN__
std::string methodToString( MR::WebRequest::Method method )
{
    using Method = MR::WebRequest::Method;
    switch ( method )
    {
        case Method::Get:
            return "GET";
        case Method::Post:
            return "POST";
        case Method::Patch:
            return "PATCH";
        case Method::Put:
            return "PUT";
        case Method::Delete:
            return "DELETE";
    }
    MR_UNREACHABLE
}
#endif

using AsyncThreads = std::unordered_map<std::thread::id, std::thread>;
AsyncThreads& getWaitingMap_()
{
    static AsyncThreads waitingMap;
    return waitingMap;
}

#ifndef __EMSCRIPTEN__
void putIntoWaitingMap_( std::thread&& thread )
{
    auto& asyncMap = getWaitingMap_();
    asyncMap[thread.get_id()] = std::move( thread );
}

#ifdef __linux__
/// ...
/// see also: https://curl.se/mail/lib-2022-05/0039.html
std::string getCaInfo( cpr::Session& session )
{
    std::error_code ec;

    // check the default CA bundle path first
    if ( auto curl = session.GetCurlHolder() )
    {
        char* caInfo = nullptr; // NOTE: the buffer should not be freed manually; see https://curl.se/libcurl/c/CURLINFO_CAINFO.html
        curl_easy_getinfo( curl->handle, CURLINFO_CAINFO, &caInfo );
        if ( caInfo )
        {
            if ( std::filesystem::is_regular_file( caInfo, ec ) )
            {
                // the default CA bundle path is valid, nothing to do
                return {};
            }
        }
    }

    // trying to find a CA bundle in known locations
    constexpr std::array cKnownCaInfoLocations {
        // Debian, Ubuntu, Arch, Alpine, ...
        "/etc/ssl/certs/ca-certificates.crt",
        // Red Hat, Fedora
        "/etc/pki/tls/certs/ca-bundle.crt",
        // some other known locations
        "/etc/ssl/ca-bundle.pem",
        "/etc/ssl/cert.pem",
        "/etc/pki/tls/cert.pem",
        "/etc/pki/tls/cacert.pem",
        "/usr/share/ssl/certs/ca-bundle.crt",
        "/usr/local/share/certs/ca-root-nss.crt",
    };
    for ( const auto& path : cKnownCaInfoLocations )
    {
        if ( std::filesystem::is_regular_file( path, ec ) )
        {
            // use the path as is, no additional check
            return path;
        }
    }

    // we did everything we could ¯\_(ツ)_/¯
    return {};
}
#endif
#endif //!__EMSCRIPTEN__

} // anonymous namespace

#ifdef __EMSCRIPTEN__
extern "C"
{

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
EMSCRIPTEN_KEEPALIVE int emsCallResponseCallback( const char* response, bool async, int ctxId )
{
    using namespace MR;
    if ( auto resJson = deserializeJsonValue( response, std::strlen( response ) ) )
    {
        if ( !async )
        {
            auto& ctx = sRequestContextMap.at( ctxId );
            ctx->responseCallback( *resJson );

            std::unique_lock lock( sRequestContextMutex );
            sRequestContextMap.erase( ctxId );
            MAIN_THREAD_EM_ASM( web_req_clear( $0 ), ctxId );
        }
        else
        {
            CommandLoop::appendCommand( [resJson, ctxId] ()
            {
                auto& ctx = sRequestContextMap.at( ctxId );
                ctx->responseCallback( *resJson );

                std::unique_lock lock( sRequestContextMutex );
                sRequestContextMap.erase( ctxId );
                MAIN_THREAD_EM_ASM( web_req_clear( $0 ), ctxId );
            } );
        }
    }
    else
    {
        spdlog::info( resJson.error() );
    }
    return 1;
}
#pragma clang diagnostic pop

EMSCRIPTEN_KEEPALIVE int emsCallUploadCallback( double v, int ctxId )
{
    auto& ctx = sRequestContextMap.at( ctxId );
    MR::reportProgress( ctx->uploadCallback, (float)v );
    return 1;
}

EMSCRIPTEN_KEEPALIVE int emsCallDownloadCallback( double v, int ctxId )
{
    auto& ctx = sRequestContextMap.at( ctxId );
    MR::reportProgress( ctx->downloadCallback, (float)v );
    return 1;
}

}
#else
namespace
{

#if CPR_VERSION_MAJOR >= 2 || ( CPR_VERSION_MAJOR == 1 && CPR_VERSION_MINOR >= 11 )
bool downloadFileCallback( const std::string_view& data, intptr_t userdata )
#else
bool downloadFileCallback( std::string data, intptr_t userdata )
#endif
{
    const auto ctxId = (int)userdata;
    auto& ctx = sRequestContextMap.at( ctxId );
    assert( ctx->output.has_value() );
    *ctx->output << data;
    return true;
}

bool progressCallback( cpr::cpr_off_t downloadTotal, cpr::cpr_off_t downloadNow, cpr::cpr_off_t uploadTotal, cpr::cpr_off_t uploadNow, intptr_t userdata )
{
    const auto ctxId = (int)userdata;
    auto& ctx = sRequestContextMap.at( ctxId );
    if ( downloadNow < downloadTotal )
        if ( !MR::reportProgress( ctx->downloadCallback, (float)downloadNow / (float)downloadTotal ) )
            return false;
    if ( uploadNow < uploadTotal )
        if ( !MR::reportProgress( ctx->uploadCallback, (float)uploadNow / (float)uploadTotal ) )
            return false;
    return true;
}

}
#endif

namespace MR
{

WebRequest::WebRequest( std::string url )
    : url_( std::move( url ) )
{
    //
}

void WebRequest::clear()
{
    method_ = Method::Get;
    url_ = {};
    logName_ = {};
    async_ = true;
    timeout_ = 10000;
    params_ = {};
    headers_ = {};
    inputPath_ = {};
    formData_ = {};
    body_ = {};
    outputPath_ = {};
    uploadCallback_ = {};
    downloadCallback_ = {};
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

void WebRequest::setInputPath( std::string inputPath )
{
    inputPath_ = std::move( inputPath );
}

void WebRequest::setUploadProgressCallback( ProgressCallback callback )
{
    uploadCallback_ = std::move( callback );
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

void WebRequest::setDownloadProgressCallback( ProgressCallback callback )
{
    downloadCallback_ = std::move( callback );
}

void WebRequest::setAsync( bool async )
{
    async_ = async;
}

void WebRequest::setLogName( std::string logName )
{
    logName_ = std::move( logName );
}

// do not pass logName by reference, since logName_ can be passed and logName_ is cleared here
void WebRequest::send( std::string urlP, std::string logName, ResponseCallback callback, bool async /*= true */ )
{
    int ctxId;
    std::shared_ptr<RequestContext> ctx;
    {
        std::unique_lock lock( sRequestContextMutex );
        ctxId = sRequestContextCounter++;
        auto [it, _] = sRequestContextMap.emplace( ctxId, std::make_shared<RequestContext>() );
        ctx = it->second;
    }

    ctx->uploadCallback = uploadCallback_;
    ctx->downloadCallback = downloadCallback_;

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

    if ( !inputPath_.empty() )
    {
        ctx->input = std::ifstream( inputPath_, std::ios::binary );
        if ( !ctx->input->good() )
        {
            spdlog::info( "WebResponse {}: Unable to open input file", logName.c_str() );
            return;
        }
    }

    if ( !outputPath_.empty() )
    {
        ctx->output = std::ofstream( outputPath_, std::ios::binary );
        if ( !ctx->output->good() )
        {
            spdlog::info( "WebResponse {}: Unable to open output file", logName.c_str() );
            return;
        }
    }

    auto sendLambda = [logName, ctxId, ctx, tm, body, params, headers, multipart, method = method_, url = urlP] () -> cpr::Response
    {
        cpr::Session session;
        session.SetUrl( url );
        session.SetHeader( headers );
        session.SetParameters( params );
        session.SetTimeout( tm );

#ifdef __linux__
        if ( url.starts_with( "https" ) )
        {
            // set the certificate info manually; see getCaInfo for more info
            static const auto cCaInfo = getCaInfo( session );
            if ( !cCaInfo.empty() )
            {
                session.SetSslOptions( cpr::Ssl(
                    cpr::ssl::CaInfo{ std::string{ cCaInfo } }
                ) );
            }
        }
#endif

        if ( ctx->input.has_value() )
        {
            // C++ Requests library doesn't support uploading from stream
            if ( auto res = readString( *ctx->input ) )
            {
                session.SetBody( std::move( *res ) );
            }
            else
            {
                spdlog::error( "WebResponse {}: Failed to read input file: {}", logName, res.error() );
            }
        }
        else if ( !multipart.parts.empty() )
        {
            session.SetMultipart( multipart );
        }
        else
        {
            session.SetBody( body );
        }

        if ( ctx->output.has_value() )
            session.SetWriteCallback( { &downloadFileCallback, ctxId } );

        if ( ctx->uploadCallback || ctx->downloadCallback )
            session.SetProgressCallback( { &progressCallback, ctxId } );

        cpr::Response r;
        switch ( method )
        {
            case Method::Get:
                r = session.Get();
                break;
            case Method::Post:
                r = session.Post();
                break;
            case Method::Patch:
                r = session.Patch();
                break;
            case Method::Put:
                r = session.Put();
                break;
            case Method::Delete:
                r = session.Delete();
                break;
        }
        spdlog::debug( "WebResponse {}: finished, updating context", logName );

        if ( ctx->output )
            ctx->output->close();
        if ( ctx->input )
            ctx->input->close();

        return r;
    };
    clear();
    if ( !async )
    {
        spdlog::info( "WebRequest  {}: {}", logName.c_str(), urlP.c_str() );
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
            spdlog::info( "WebRequest  {}: {}", logName.c_str(), url.c_str() );
            auto res = sendLambda();

            // log everything in one line for convenience
            std::string info = "status_code=" + std::to_string( res.status_code );
            if ( !res.status_line.empty() )
                info += ", status_line=" + res.status_line;
            if ( !res.reason.empty() )
                info += ", reason=" + res.reason;
            if ( res.error )
            {
                info += ", error_code=" + std::to_string( int( res.error.code ) );
                info += ", error_message=" + res.error.message;
            }
            spdlog::info( "WebResponse {}: {}", logName.c_str(), info );

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
        putIntoWaitingMap_( std::move( requestThread ) );
    }
#else
    (void)logName;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    const auto method = methodToString( method_ );
    MAIN_THREAD_EM_ASM( {
        create_web_ctx_if_needed($6);
        web_req_ctxs[$6].timeout = $0;
        web_req_ctxs[$6].body = UTF8ToString( $1 );
        web_req_ctxs[$6].filename = UTF8ToString( $2 );
        web_req_ctxs[$6].method = UTF8ToString( $3 );
        web_req_ctxs[$6].use_upload_callback = $4;
        web_req_ctxs[$6].use_download_callback = $5;
    },
        timeout_,
        body_.c_str(),
        inputPath_.c_str(),
        method.c_str(),
        (bool)uploadCallback_,
        (bool)downloadCallback_,
        ctxId
    );

    for ( const auto& [key, value] : params_ )
        MAIN_THREAD_EM_ASM( web_req_add_param( UTF8ToString( $0 ), UTF8ToString( $1 ), $2 ), key.c_str(), value.c_str(), ctxId );

    for ( const auto& [key, value] : headers_ )
        MAIN_THREAD_EM_ASM( web_req_add_header( UTF8ToString( $0 ), UTF8ToString( $1 ), $2 ), key.c_str(), value.c_str(), ctxId );

    for ( const auto& formData : formData_ )
    {
        MAIN_THREAD_EM_ASM(
            web_req_add_formdata( UTF8ToString( $0 ), UTF8ToString( $1 ), UTF8ToString( $2 ), UTF8ToString( $3 ), $4 ),
            formData.path.c_str(),
            formData.contentType.c_str(),
            formData.name.c_str(),
            formData.fileName.c_str(),
            ctxId
        );
    }

    if ( !urlP.empty() && urlP.back() == '/' )
        urlP = urlP.substr( 0, int( urlP.size() ) - 1 );

    ctx->responseCallback = callback;

    if ( outputPath_.empty() )
        MAIN_THREAD_EM_ASM( web_req_send( UTF8ToString( $0 ), $1, $2 ), urlP.c_str(), async, ctxId );
    else
        MAIN_THREAD_EM_ASM( web_req_async_download( UTF8ToString( $0 ), UTF8ToString( $1 ), $2 ), urlP.c_str(), outputPath_.c_str(), ctxId );
#pragma clang diagnostic pop
#endif
}

void WebRequest::send( WebRequest::ResponseCallback callback )
{
    if ( url_.empty() )
    {
        spdlog::warn( "WebRequest {}: URL is not specified", logName_ );
        return;
    }

    send( url_, logName_, std::move( callback ), async_ );
}

void WebRequest::waitRemainingAsync()
{
    auto& asyncMap = getWaitingMap_();
    for ( auto& [_, thread] : asyncMap )
        if ( thread.joinable() )
            thread.join();
}

Expected<Json::Value> parseResponse( const Json::Value& response )
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

    auto rootRes = deserializeJsonValue( text );
    if ( !rootRes )
        return unexpected( "Unknown error." );
    auto& root = *rootRes;

    if ( root.isObject() && root["message"].isString() )
        return unexpected( root["message"].asString() );
    return root;
}

}
#endif
