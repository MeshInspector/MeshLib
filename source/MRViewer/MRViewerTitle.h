#pragma once
#include "MRViewerFwd.h"
#include <string>

namespace MR
{

/// class to compose viewer title from several parts
/// appName (version) sceneName
class MRVIEWER_CLASS ViewerTitle
{
public:
    ViewerTitle() = default;
    virtual ~ViewerTitle() = default;

    /// Name of the application
    MRVIEWER_API void setAppName( std::string appName );

    /// Version of application
    MRVIEWER_API void setVersion( std::string version );

    /// Name of current scene
    MRVIEWER_API void setSceneName( std::string sceneName );

    // this function is called from `update_` compose resulting name
    MRVIEWER_API virtual std::string compose() const;
protected:
    // called when something changes, composes new title and apply it to window
    MRVIEWER_API void update_();

    std::string appName_;
    std::string version_;
    std::string sceneName_;

private:
    std::string composed_;
};

}