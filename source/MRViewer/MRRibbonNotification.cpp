#include "MRRibbonNotification.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include <imgui.h>

namespace
{
constexpr int cNotificationNumberLimit = 5;
}

namespace MR
{

void RibbonNotificationDrawer::pushNotification( const RibbonNotification& notification )
{
    if ( notifications_.size() == cNotificationNumberLimit )
        notifications_.erase( notifications_.end() - 1 );
    notifications_.insert( notifications_.begin(), NotificationWithTimer{ notification } );
    requestClosestRedraw_();
}

void RibbonNotificationDrawer::drawNotifications( float scaling )
{
    for ( int i = 0; i < notifications_.size(); ++i )
    {

    }
    filterInvalid_();
}

void RibbonNotificationDrawer::filterInvalid_()
{
    bool changed = false;
    for ( int i = int( notifications_.size() ) - 1; i >= 0; ++i )
    {
        if ( notifications_[i].notification.lifeTimeSec - notifications_[i].timer <= 0.0f )
        {
            changed = true;
            notifications_.erase( notifications_.begin() + i );
        }
    }
    if ( changed )
        requestClosestRedraw_();
}

void RibbonNotificationDrawer::requestClosestRedraw_()
{
    float minTimeReq = FLT_MAX;
    for ( const auto& notification : notifications_ )
    {
        float neededTime = notification.notification.lifeTimeSec - notification.timer;
        if ( neededTime < minTimeReq )
            minTimeReq = neededTime;
    }
    if ( minTimeReq == FLT_MAX )
        return;
#ifndef __EMSCRIPTEN__
    Time neededTime = std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( minTimeReq * 1000 ) + 200 );
    if ( requrestedTime_ < neededTime )
        return;
    requrestedTime_ = neededTime;
    asyncRequest_.request( requrestedTime_, [] ()
    {
        CommandLoop::appendCommand( [] ()
        {
            getViewerInstance().incrementForceRedrawFrames();
        } );
    } );
#endif
}

}