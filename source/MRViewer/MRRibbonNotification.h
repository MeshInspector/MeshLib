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
    // negative value means to use default one
    float lifeTimeSec = -1.0f;
    // if notifications are equal to last one added, it just increment counter
    // note that if there is present `onButtonClick` this function always returns false
    bool operator==( const RibbonNotification& other ) const;
};

// class to hold and operate with notifications
class MRVIEWER_CLASS RibbonNotifier
{
public:
    // adds new notification for drawing
    // note that `lifeTimeSec` will be overridden with `defaultNotificationLifeTimeSeconds` even while `notification` is const
    MRVIEWER_API void pushNotification( const RibbonNotification& notification );
    // main draw function. draw actual notification or history, and history button
    MRVIEWER_API void draw( float scaling, float scenePosX, float topPanelHeight );

    // this value is used as notification `lifeTimeSec` if negative values passed
    float defaultNotificationLifeTimeSeconds = 5.0f;
private:
    struct NotificationWithTimer
    {
        RibbonNotification notification;
        float timer{ 0.0f };
        int sameCounter = 1;
    };
    std::vector<NotificationWithTimer> notifications_;
    std::vector<NotificationWithTimer> notificationsHistory_;
    bool requestRedraw_ = false;
    bool historyMode_ = false;

#ifndef __EMSCRIPTEN__
    Time requestedTime_{ Time::max() };
    AsyncRequest asyncRequest_;
#endif

    // draw button to show last notifications
    void drawHistoryButton_( float scaling, float scenePosX );
    // draw notification history
    void drawHistory_( float scaling, float scenePosX, float topPanelHeight );
    // draw floating notifications
    void drawFloating_( float scaling, float scenePosX );
    
    // set this true on open history and on new notification added
    bool scrollDownNeeded_ = false;
    float prevHistoryScrollMax_ = 0.0f;
    struct DrawNotificationSettings
    {
        int index{ 0 };
        float scalig{ 1.0f };
        float width{ 0.0f };
        bool historyMode{ false };
        Vector2f* currentPos{ nullptr };
    };
    // draws one notification
    // returns false if need to close
    bool drawNotification_( const DrawNotificationSettings& settings );
    void addNotification_( std::vector<NotificationWithTimer>& store, const RibbonNotification& notification );
    void filterInvalid_( int numInvalid = -1 );
    void requestClosestRedraw_();
};

}
