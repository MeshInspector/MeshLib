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
class MRVIEWER_CLASS RibbonNotifier
{
public:
    // adds new notification for drawing
    void pushNotification( const RibbonNotification& notification );
    // main draw function. draw actual notification or history, and history button
    void draw( float scaling, float scenePosX );
private:
    struct NotificationWithTimer
    {
        RibbonNotification notification;
        float timer{ 0.0f };
        int sameCounter = 1;
    };
    std::vector<NotificationWithTimer> notifications_;
    std::vector<NotificationWithTimer> notificationsHistory_;
    NotificationType highestNotification_ = NotificationType::Count;
    bool requestRedraw_ = false;
    bool historyMode_ = false;

#ifndef __EMSCRIPTEN__
    Time requestedTime_{ Time::max() };
    AsyncRequest asyncRequest_;
#endif

    // draw button to show last notifications
    void drawHistoryButton_( float scaling, float scenePosX );
    // draw notification history
    void drawHistory_( float scaling, float scenePosX );
    // draws all present notifications
    void drawNotifications_( float scaling, float scenePosX );
    void addNotification_( std::vector<NotificationWithTimer>& store, const RibbonNotification& notification );
    void filterInvalid_( int numInvalid = -1 );
    void requestClosestRedraw_();
};

}
