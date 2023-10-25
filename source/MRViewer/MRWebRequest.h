#pragma once
#include "MRMesh/MRMeshFwd.h"
#ifndef MRMESH_NO_CPR
#include "MRViewerFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRPch/MRJson.h"
#include <unordered_map>
#include <string>
#include <functional>

namespace MR
{
// returns json value of text or error if response failed
MRVIEWER_API Expected<Json::Value, std::string> parseResponse( const Json::Value& response );

// this class is needed to unify cpp and wasm requests
// should be called from GUI thread
class MRVIEWER_CLASS WebRequest
{
public:
    enum class Method 
    {
        Get,
        Post
    };

    // clears all request data
    MRVIEWER_API void clear();

    MRVIEWER_API void setMethod( Method method );

    // sets timeout in milliseconds
    MRVIEWER_API void setTimeout( int timeoutMs );

    // sets parameters
    MRVIEWER_API void setParameters( std::unordered_map<std::string, std::string> parameters );

    MRVIEWER_API void setHeaders( std::unordered_map<std::string, std::string> headers );

    MRVIEWER_API void setBody( std::string body );

    using ResponseCallback = std::function<void( const Json::Value& response )>;

    /// sends request, calling callback on answer, 
    /// if async then callback is called in next frame after getting response
    /// return true if request was sent, false if other request is processing now
    /// note: check `readyForNextRequest` before sending
    /// \param logName name for logging
    MRVIEWER_API void send( std::string url, const std::string & logName, ResponseCallback callback, bool async = true );

private:
    Method method_{ Method::Get };
    int timeout_{ 10000 };
    std::unordered_map<std::string, std::string> params_;
    std::unordered_map<std::string, std::string> headers_;
    std::string body_;
};

}
#endif
