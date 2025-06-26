#pragma once
#include "MRMesh/MRMeshFwd.h"
#if defined( __EMSCRIPTEN__ ) || !defined( MRMESH_NO_CPR )
#include "MRViewerFwd.h"
#include "MRMesh/MRExpected.h"
#include <json/forwards.h>
#include <unordered_map>
#include <thread>
#include <string>
#include <functional>

namespace MR
{
// returns json value of text or error if response failed
MRVIEWER_API Expected<Json::Value> parseResponse( const Json::Value& response );

// this class is needed to unify cpp and wasm requests
class MRVIEWER_CLASS WebRequest
{
public:
    WebRequest() = default;
    MRVIEWER_API explicit WebRequest( std::string url );

    enum class Method 
    {
        Get,
        Post,
        Patch,
        Put,
        Delete,
    };

    // clear all request data
    MRVIEWER_API void clear();

    // set HTTP method
    MRVIEWER_API void setMethod( Method method );

    // set timeout in milliseconds
    MRVIEWER_API void setTimeout( int timeoutMs );

    // set URL query parameters
    MRVIEWER_API void setParameters( std::unordered_map<std::string, std::string> parameters );

    // set HTTP headers
    MRVIEWER_API void setHeaders( std::unordered_map<std::string, std::string> headers );

    // set path to the file to upload
    MRVIEWER_API void setInputPath( std::string inputPath );

    // set progress callback for upload
    // NOTE: due to limitations, the upload callback won't work on web platforms when `setOutputPath` method is called
    MRVIEWER_API void setUploadProgressCallback( ProgressCallback callback );

    // set payload in multipart/form-data format
    struct FormData
    {
        std::string path;
        std::string contentType;
        std::string name;
        std::string fileName;
    };
    MRVIEWER_API void setFormData( std::vector<FormData> formData );

    // set payload in plain format
    MRVIEWER_API void setBody( std::string body );

    // prefer to save the response to file
    MRVIEWER_API void setOutputPath( std::string outputPath );

    // set progress callback for download
    MRVIEWER_API void setDownloadProgressCallback( ProgressCallback callback );

    // set async mode (return immediately after sending request)
    MRVIEWER_API void setAsync( bool async );

    // set log name
    MRVIEWER_API void setLogName( std::string logName );

    using ResponseCallback = std::function<void( const Json::Value& response )>;

    /// send request, calling callback on answer,
    /// if async then callback is called in next frame after getting response
    /// NOTE: downloading a binary file in synchronous mode is forbidden by JavaScript
    /// \param logName name for logging
    MRVIEWER_API void send( std::string url, std::string logName, ResponseCallback callback, bool async = true );
    MRVIEWER_API void send( ResponseCallback callback );

    /// if any async request is still in progress, wait for it
    MRVIEWER_API static void waitRemainingAsync();
private:
    Method method_{ Method::Get };
    std::string url_;
    std::string logName_;
    bool async_{ true };
    int timeout_{ 10000 };
    std::unordered_map<std::string, std::string> params_;
    std::unordered_map<std::string, std::string> headers_;
    std::string inputPath_;
    std::vector<FormData> formData_;
    std::string body_;
    std::string outputPath_;
    ProgressCallback uploadCallback_;
    ProgressCallback downloadCallback_;

    using AsyncThreads = std::unordered_map<std::thread::id, std::thread>;
    static AsyncThreads& getWaitingMap_();
#ifndef __EMSCRIPTEN__
    void putIntoWaitingMap_( std::thread&& thread );
#endif
};

}
#endif
