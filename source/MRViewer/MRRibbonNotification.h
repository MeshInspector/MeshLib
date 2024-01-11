#pragma once
#include "MRViewerFwd.h"
#include "MRNotificationType.h"
#include "MRAsyncTimer.h"
#include <functional>
#include <chrono>

namespace MR
{
struct RibbonNotification
{
    // Type of notification
    NotificationType type{ NotificationType::Info };
    // Callback for notification
    // return true to stay open, false to close
    // args: float windowWidth, float menuScaling
    using DrawContentFunc = std::function<bool( float, float )>;
    DrawContentFunc drawContentFunc;
    // Time that notification stays visible
    float lifeTimeSec = 2.0f;
};

// class to hold and operate with notifications
class RibbonNotificationDrawer
{
public:
    // adds new notification for drawing
    void pushNotification( const RibbonNotification& notification );
    // draws all present notifications
    void drawNotifications( float scaling );
private:
    struct NotificationWithTimer
    {
        RibbonNotification notification;
        float timer{ 0.0f };
    };
    std::vector<NotificationWithTimer> notifications_;
    void filterInvalid_();
#ifndef __EMSCRIPTEN__
    Time requrestedTime_;
    AsyncRequest asyncRequest_;
#endif
    void requestClosestRedraw_();
};

}