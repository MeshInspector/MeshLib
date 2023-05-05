#pragma once
#ifndef MRMESH_NO_CPR
#include "MRViewerFwd.h"
#include "MRPch/MRJson.h"
#include <tl/expected.hpp>
#include <unordered_map>
#include <string>
#include <functional>

namespace MR
{
// returns json value of text or error if response failed
MRVIEWER_API tl::expected<Json::Value, std::string> parseResponse( const Json::Value& response );

// this class is needed to unify cpp and wasm requests
// can perform only one request at a time
// should be called from GUI thread
class MRVIEWER_CLASS WebRequest
{
public:
    enum class Method 
    {
        Get,
        Post
    };

    // returns true if no other request is executing right now
    MRVIEWER_API static bool readyForNextRequest();

    // clears all request data
    MRVIEWER_API static void clear();

    MRVIEWER_API static void setMethod( Method method );

    // sets timeout in milliseconds
    MRVIEWER_API static void setTimeout( int timeoutMs );

    // sets parameters
    MRVIEWER_API static void setParameters( std::unordered_map<std::string, std::string> parameters );

    MRVIEWER_API static void setHeaders( std::unordered_map<std::string, std::string> headers );

    MRVIEWER_API static void setBody( std::string body );

    using ResponseCallback = std::function<void( const Json::Value& response )>;

    // sends request, calling callback on answer, 
    // if async then callback is called in next frame after getting response
    // return true if request was sent, false if other request is processing now
    // note: check `readyForNextRequest` before sending
    MRVIEWER_API static bool send( std::string url, ResponseCallback callback, bool async = true );
private:
    WebRequest() = default;
    ~WebRequest() = default;

    Method method_{ Method::Get };
    int timeout_{ 10000 };
    std::unordered_map<std::string, std::string> params_;
    std::unordered_map<std::string, std::string> headers_;
    std::string body_;

    static WebRequest& instance_();
};

}
#endif
