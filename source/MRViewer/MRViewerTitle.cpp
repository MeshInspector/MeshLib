#include "MRViewerTitle.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{

void ViewerTitle::setAppName( std::string appName )
{
    if ( appName == appName_ )
        return;
    appName_ = std::move( appName );
    update_();
}

void ViewerTitle::setVersion( std::string version )
{
    if ( version == version_ )
        return;
    version_ = std::move( version );
    update_();
}

void ViewerTitle::setSceneName( std::string sceneName )
{
    if ( sceneName == sceneName_ )
        return;
    sceneName_ = std::move( sceneName );
    update_();
}

std::string ViewerTitle::compose() const
{
    std::string res;
    if ( !appName_.empty() )
        res += appName_;
    if ( !version_.empty() )
    {
        if ( !res.empty() )
            res += " ";
        res += "(" + version_ + ")";
    }
    if ( !sceneName_.empty() )
    {
        if ( !res.empty() )
            res += " ";
        res += sceneName_;
    }
    return res;
}

void ViewerTitle::update_()
{
    composed_ = compose();
    auto* window = getViewerInstance().window;
    if ( window )
        glfwSetWindowTitle( window, composed_.c_str() );
}

}