#pragma once
#include "MRViewerFwd.h"
#include "MRNotificationType.h"
#include "MRAsyncTimer.h"
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRBox.h"
#include <functional>
#include <chrono>

namespace MR
{

struct NotificationTags
{
    enum Tag : unsigned
    {
        None = 0b0000,
        Report = 0b0001,
        Recommendation = 0b0010,
        ImplicitChanges = 0b0100,
        Important = 0b1000,
        Default = Important | ImplicitChanges,
        All = Report | ImplicitChanges | Recommendation | Important,
    };
};
MR_MAKE_FLAG_OPERATORS( NotificationTags::Tag )

using NotificationTagMask = unsigned;

/// corner where notifications will appear
enum class RibbonNotificationCorner
{
    LowerLeft,
    LowerRight
};

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
    // it ANDs with RibbonNotifier allowed tags to see if notification should be displayed
    NotificationTagMask tags = NotificationTags::All;
    // if notifications are equal to last one added, it just increment counter
    // note that if there is present `onButtonClick` this function always returns false
    bool operator==( const RibbonNotification& other ) const;
};

// class to hold and operate with notifications
class MRVIEWER_CLASS RibbonNotifier
{
public:
    // adds new notification for drawing
    MRVIEWER_API void pushNotification( const RibbonNotification& notification );

    // main draw function. draw actual notification or history, and history button
    // limitFramebuffer - available framebuffer space (usually same as `Viewer::getViewportsBounds()`)
    MRVIEWER_API void draw( const Box2i& limitFramebuffer );

    // set maximum time while history button will be present on screen
    // negative value means that history button will never be hidden
    MRVIEWER_API void setHitoryButtonMaxLifeTime( float histBtnMaxLifeTime );

    // this value is used as notification `lifeTimeSec` if negative values passed
    float defaultNotificationLifeTimeSeconds = 5.0f;

    // this mask is used to control allowed notifications by filtering with tags
    NotificationTagMask allowedTagMask = NotificationTags::Default;

    // position of notifications on screen
    RibbonNotificationCorner cornerPosition = RibbonNotificationCorner::LowerLeft;
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

    float showHistoryBtnMaxTime_{ -1.0f }; // negative value here means that there is no need to hide history button
    float currentHistoryBtnTimer_{ -1.0f }; // update to validly hide the button
#ifndef __EMSCRIPTEN__
    Time requestedTime_{ Time::max() };
    AsyncRequest asyncRequest_;
#endif

    // draw button to show last notifications
    void drawHistoryButton_( const Box2i& limitFramebuffer );
    // draw notification history
    void drawHistory_( const Box2i& limitFramebuffer );
    // draw floating notifications
    void drawFloating_( const Box2i& limitFramebuffer );

    // set this true on open history and on new notification added
    bool scrollDownNeeded_ = false;
    float prevHistoryScrollMax_ = 0.0f;
    struct DrawNotificationSettings
    {
        int index{ 0 };
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
