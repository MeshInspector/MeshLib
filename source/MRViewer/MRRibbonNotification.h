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
    // Callback for notification
    // if it is not null, a button will be drawn, and callback will be invoked on button click
    using OnButtonClick = std::function<void()>;
    OnButtonClick onButtonClick;

    // Name of button that will be drawn if callback is enabled
    std::string buttonName = "OK";
    // Header of notification
    std::string header;
    // Text of notification
    std::string text;
    // Type of notification
    NotificationType type{ NotificationType::Info };
    // Time that notification stays visible
    float lifeTimeSec = 10.0f;
    // if notifications are equal to last one added, it just increment counter
    // note that if there is present `onButtonClick` this function always returns false
    bool operator==( const RibbonNotification& other ) const;
};

// class to hold and operate with notifications
class RibbonNotifier
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
        int sameCounter = 1;
    };
    std::vector<NotificationWithTimer> notifications_;
    void filterInvalid_( int numInvalid = -1 );
#ifndef __EMSCRIPTEN__
    Time requestedTime_{ Time::max() };
    AsyncRequest asyncRequest_;
#endif
    void requestClosestRedraw_();
};

}