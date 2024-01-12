#include "MRRibbonNotification.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "ImGuiHelpers.h"
#include <imgui.h>
#include <imgui_internal.h>

namespace
{
constexpr int cNotificationNumberLimit = 10;
}

namespace MR
{

void RibbonNotifier::pushNotification( const RibbonNotification& notification )
{
    if ( notifications_.size() == cNotificationNumberLimit )
        notifications_.erase( notifications_.end() - 1 );
    notifications_.insert( notifications_.begin(), NotificationWithTimer{ notification } );
    requestClosestRedraw_();
}

void RibbonNotifier::drawNotifications( float scaling )
{
    Vector2f currentPos = Vector2f( getViewerInstance().framebufferSize );
    const Vector2f padding = Vector2f( 0.0f, 20.0f * scaling );
    const float width = 250.0f * scaling;
    currentPos.x -= padding.y;

    int numInvalid = -1;
    for ( int i = 0; i < notifications_.size(); ++i )
    {
        currentPos -= padding;
        auto& [notification, timer] = notifications_[i];

        ImGui::SetNextWindowPos( currentPos, ImGuiCond_Always, ImVec2( 1.0f, 1.0f ) );
        ImGui::SetNextWindowSize( ImVec2( width, -1 ), ImGuiCond_Always );
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoMove;
        std::string name = "##notification" + std::to_string( i );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 10.0f * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 15.0f * scaling, 15.0f * scaling ) );
        if ( i + 1 == cNotificationNumberLimit )
            ImGui::SetNextWindowBgAlpha( 0.7f );
        ImGui::Begin( name.c_str(), nullptr, flags );
        if ( notification.drawContentFunc )
            if ( !notification.drawContentFunc( width, scaling ) )
                numInvalid = i;
        auto window = ImGui::GetCurrentContext()->CurrentWindow;
        ImGui::End();
        ImGui::PopStyleVar( 3 );
        if ( !ImGui::IsWindowHovered() )
            timer += ImGui::GetIO().DeltaTime;
        currentPos.y -= window->Size.y;
    }
    filterInvalid_( numInvalid );
}

void RibbonNotifier::filterInvalid_( int numInvalid )
{
    bool changed = false;
    for ( int i = int( notifications_.size() ) - 1; i >= 0; --i )
    {
        if ( notifications_[i].notification.lifeTimeSec - notifications_[i].timer <= 0.0f || i == numInvalid )
        {
            changed = true;
            notifications_.erase( notifications_.begin() + i );
        }
    }
    if ( changed )
    {
        requrestedTime_ = Time::max();
        requestClosestRedraw_();
    }
}

void RibbonNotifier::requestClosestRedraw_()
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
    Time neededTime = std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( minTimeReq * 1000 ) + 100 );
    if ( requrestedTime_ < neededTime )
        return;
    requrestedTime_ = neededTime;
    asyncRequest_.request( requrestedTime_, [&] ()
    {
        CommandLoop::appendCommand( [&] ()
        {
            getViewerInstance().incrementForceRedrawFrames();
            requrestedTime_ = Time::max();
        } );
    } );
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( postEmptyEvent( $0, 2 ), int( minTimeReq * 1000 ) + 100 );
#pragma clang diagnostic pop
#endif
}

}