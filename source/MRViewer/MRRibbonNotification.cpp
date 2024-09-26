#include "MRRibbonNotification.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "ImGuiHelpers.h"
#include "MRMesh/MRColor.h"
#include "MRPch/MRWasm.h"
#include "MRProgressBar.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRRibbonMenu.h"
#include "MRUITestEngine.h"
#include <imgui_internal.h>

namespace
{
constexpr int cNotificationNumberLimit = 10;

constexpr std::array< std::pair<const char*, ImU32>, int( MR::NotificationType::Count )> notificationParams
{
    std::pair<const char*, ImU32> { "\xef\x81\xaa", 0xff4444e2 },
    std::pair<const char*, ImU32> { "\xef\x81\xb1", 0xff0092ff },
    std::pair<const char*, ImU32> { "\xef\x83\xb3", 0xffff831b },
    std::pair<const char*, ImU32> { "\xef\x8b\xb2", 0xff0092ff }
};


}

namespace MR
{

void RibbonNotifier::pushNotification( const RibbonNotification& notification )
{
    if ( !historyMode_ )
        addNotification_( notifications_, notification );
    addNotification_( notificationsHistory_, notification );

    highestNotification_ = NotificationType::Count;
    for ( const auto& nwt : notificationsHistory_ )
    {
        if ( int( nwt.notification.type ) < int( highestNotification_ ) )
            highestNotification_ = nwt.notification.type;
    }
    
    requestClosestRedraw_();
}

void RibbonNotifier::draw( float scaling, float scenePosX )
{
    if ( historyMode_ )
        drawHistory_( scaling, scenePosX );
    else
        drawNotifications_( scaling, scenePosX );
    drawHistoryButton_( scaling, scenePosX );
    filterInvalid_( -1 );
}

void RibbonNotifier::drawHistoryButton_( float scaling, float scenePosX )
{
    using namespace StyleConsts::Notification;
    if ( notificationsHistory_.empty() )
        return;

    float notificationsPosX = cWindowSpacing * scaling + scenePosX;
    Vector2f windowPos = Vector2f( notificationsPosX, float( getViewerInstance().framebufferSize.y ) - 40.f * scaling );
    const float windowPadding = cWindowPadding * scaling;

    
    auto iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( iconsFont )
    {
        iconsFont->Scale = 0.7f;
        ImGui::PushFont( iconsFont );
    }
    const float size = ImGui::GetTextLineHeight() + windowPadding * 2.f;
    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always, ImVec2(0.f, 1.0f));
    ImGui::SetNextWindowSize( ImVec2( size, size ), ImGuiCond_Always );
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoMove;
    std::string name = "##NotificationButton";
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( windowPadding, windowPadding ) );
    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, { 0, 0 } );
    ImGui::PushStyleColor( ImGuiCol_WindowBg, MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::FrameBackground ).getUInt32() );

    ImGui::Begin( name.c_str(), nullptr, flags );

    ImGui::PushStyleColor( ImGuiCol_Text, notificationParams[int( highestNotification_ )].second );
    ImGui::Text( "%s", notificationParams[int( highestNotification_ )].first );
    ImGui::PopStyleColor();

    if ( iconsFont )
    {
        iconsFont->Scale = 1.0f;
        ImGui::PopFont();
    }

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    bool isHovered = ImGui::IsWindowHovered();

    if ( isHovered )
    {
        if ( ImGui::IsMouseClicked( ImGuiMouseButton_Left ) )
        {
            historyMode_ = !historyMode_;
            if ( historyMode_ )
                notifications_.clear();
        }

        auto drawList = window->DrawList;
        drawList->PushClipRectFullScreen();
        const ImU32 color = ImGui::GetColorU32( ImGuiCol_Text );
        drawList->AddRect( window->Rect().Min, window->Rect().Max, color, cWindowRounding * scaling, 0, cWindowBorderWidth * scaling );
        drawList->PopClipRect();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 4 );
}

void RibbonNotifier::drawHistory_( float scaling, float scenePosX )
{
    using namespace StyleConsts::Notification;
    const float cWindowExpansion = cWindowPadding;

    float windowPosX = ( cWindowSpacing  - cWindowExpansion ) * scaling + scenePosX;
    const float windowPosShiftY = ( cWindowsPosY + cWindowExpansion ) * scaling;
    Vector2f windowPos = Vector2f( windowPosX, float( getViewerInstance().framebufferSize.y ) - windowPosShiftY );
    const float width = ( 337.0f + cWindowExpansion * 2 )* scaling;

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always, ImVec2( 0.f, 1.0f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( width, 1 ), ImVec2( width, float( getViewerInstance().framebufferSize.y ) - windowPosShiftY ) );
    ImGui::SetNextWindowSize( ImVec2( width, -1 ), ImGuiCond_Always );
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoMove;
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( ( cNotificationWindowPaddingX + cWindowExpansion ) * scaling,
        ( cNotificationWindowPaddingY + cWindowExpansion ) * scaling ) );
    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, { 0, 0 } );
    ImGui::PushStyleColor( ImGuiCol_WindowBg, MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::FrameBackground ).getUInt32() );
    ImGui::Begin( "NotificationsHistory", nullptr, flags );

    const float firstColumnWidth = 28.0f * scaling;
    ImGui::BeginTable( "##NotificationTable", 2, ImGuiTableFlags_SizingFixedFit );
    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, firstColumnWidth );
    ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, width - firstColumnWidth );
    const int notificationsHistorySize = int( notificationsHistory_.size() );
    for ( int i = 0; i < notificationsHistorySize; ++i )
    {
        const float beginCursorPosY = ImGui::GetCursorPosY();
        auto& [notification, timer, counter] = notificationsHistory_[notificationsHistorySize - 1 - i];
        auto& style = ImGui::GetStyle();

        ImGui::TableNextColumn();
        auto iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
        if ( iconsFont )
        {
            iconsFont->Scale = 0.7f;
            ImGui::PushFont( iconsFont );
        }

        ImGui::PushStyleColor( ImGuiCol_Text, notificationParams[int( notification.type )].second );
        if ( notification.onButtonClick )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
        ImGui::Text( "%s", notificationParams[int( notification.type )].first );
        ImGui::PopStyleColor();

        if ( iconsFont )
        {
            iconsFont->Scale = 1.0f;
            ImGui::PopFont();
        }
        
        ImGui::TableNextColumn();

        if ( !notification.header.empty() )
        {
            auto boldFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::BigSemiBold );
            if ( boldFont )
                ImGui::PushFont( boldFont );

            ImGui::SetCursorPosX( 50.0f * scaling );
            if ( notification.onButtonClick )
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
            ImGui::TextWrapped( "%s", notification.header.c_str() );

            if ( boldFont )
                ImGui::PopFont();
        }

        auto bigFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Big );
        if ( bigFont )
            ImGui::PushFont( bigFont );

        ImGui::SetCursorPosX( 50.0f * scaling );
        if ( notification.onButtonClick )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
        ImGui::TextWrapped( "%s", notification.text.c_str() );

        if ( bigFont )
            ImGui::PopFont();

        const ImU32 color = notificationParams[int( notification.type )].second;
        auto drawList = window->DrawList;
        auto windRect = window->Rect();
        ImRect rect;
        ImVec2 size = ImVec2( window->Size.x - cWindowExpansion * 2 * scaling, ImGui::GetCursorPosY() - beginCursorPosY - ImGui::GetStyle().ItemSpacing.y + cWindowPadding * 2 * scaling );
        rect.Min.x = windRect.Min.x + cWindowExpansion * scaling;
        rect.Min.y = windRect.Min.y + beginCursorPosY - cWindowPadding * scaling - ImGui::GetScrollY();
        rect.Max = rect.Min + size;
        drawList->PushClipRect( windRect.Min, windRect.Max );
        drawList->AddRect( rect.Min, rect.Max, color, cWindowRounding * scaling, 0, cWindowBorderWidth * scaling );
        drawList->PopClipRect();

        if ( counter > 1 )
        {
            const ImU32 textColor = ImGui::GetColorU32( ImGuiCol_WindowBg );
            auto countText = std::to_string( counter );
            auto textWidth = ImGui::CalcTextSize( countText.c_str() ).x;
            size = ImVec2( textWidth + ImGui::GetStyle().FramePadding.x * 2, ImGui::GetFrameHeight() );
            rect.Min.x = windRect.Max.x - cWindowExpansion * scaling;
            rect.Min.y = windRect.Min.y + ImGui::GetCursorPosY() - ImGui::GetScrollY();
            rect.Min -= size * 0.5f;
            rect.Max = rect.Min + size;
            drawList->PushClipRectFullScreen();
            drawList->AddRectFilled( rect.Min, rect.Max, color, cWindowRounding * scaling, 0 );
            drawList->AddText( rect.Min + ImGui::GetStyle().FramePadding, textColor, countText.c_str() );
            drawList->PopClipRect();
        }

        if ( i + 1 < notificationsHistory_.size() )
        {
            ImGui::TableNextRow( ImGuiTableRowFlags_None, ( cWindowSpacing + cNotificationWindowPaddingY * 2.f ) * scaling );
            ImGui::TableNextRow();
        }
    }
    ImGui::EndTable();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 4 );
}

void RibbonNotifier::drawNotifications_( float scaling, float scenePosX )
{
    using namespace StyleConsts::Notification;
    float notificationsPosX = scenePosX;
    Vector2f currentPos = Vector2f( notificationsPosX, float ( getViewerInstance().framebufferSize.y ) - cWindowsPosY * scaling );
    const Vector2f padding = Vector2f( 0.0f, cWindowSpacing * scaling );
    const float width = 319.0f * scaling;
    currentPos.x += padding.y;

    int filteredNumber = -1;
    for ( int i = 0; i < notifications_.size(); ++i )
    {
        currentPos -= padding;
        auto& [notification, timer,counter] = notifications_[i];

        ImGui::SetNextWindowPos( currentPos, ImGuiCond_Always, ImVec2( 0.f, 1.0f ) );
        ImGui::SetNextWindowSize( ImVec2( width, -1 ), ImGuiCond_Always );
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoMove;
        std::string name = "##notification" + std::to_string( i );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cNotificationWindowPaddingX * scaling, cNotificationWindowPaddingY * scaling ) );
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, { 0, 0 } );
        ImGui::PushStyleColor( ImGuiCol_WindowBg, MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::FrameBackground ).scaledAlpha( 0.6f ).getUInt32() );

        auto activeModal = ImGui::GetTopMostPopupModal();

        if ( i + 1 == cNotificationNumberLimit )
            ImGui::SetNextWindowBgAlpha( 0.5f );
        ImGui::Begin( name.c_str(), nullptr, flags );

        auto window = ImGui::GetCurrentContext()->CurrentWindow;

        if ( ImGui::IsWindowAppearing() )
        {
            if ( !activeModal || std::string_view( activeModal->Name ) != " Error##modal" )
                ImGui::BringWindowToDisplayFront( window ); // bring to front to be over modal background (but not over menu modal)

            if ( !ProgressBar::isOrdered() && !activeModal ) // do not focus window, not to close modal on appearing
                ImGui::SetWindowFocus();
        }

        auto& style = ImGui::GetStyle();
        const float buttonWidth = notification.onButtonClick ?
            ImGui::CalcTextSize( notification.buttonName.c_str() ).x + 2.0f * style.FramePadding.x + 2.0f * style.WindowPadding.x : 0;

        auto drawList = window->DrawList;
        auto cirlcePos = ImGui::GetCursorScreenPos();
        const auto radius = 3 * scaling;
        const auto bigFontSize = RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::SemiBold ) * scaling;
        cirlcePos.x += radius;
        cirlcePos.y += bigFontSize * 0.5f + radius;
        drawList->AddCircleFilled( cirlcePos, 3.0f * scaling, notificationParams[int( notification.type )].second );

        const ImVec2 contentShift = ImVec2( 26.0f * scaling, radius );
        if ( !notification.header.empty() )
        {
            auto boldFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::SemiBold );
            if ( boldFont )
                ImGui::PushFont( boldFont );
            
            ImGui::SetCursorPosX( contentShift.x );
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + contentShift.y );
            bool changeHeaderColor = notification.type == NotificationType::Error || notification.type == NotificationType::Warning;
            if ( changeHeaderColor )
                ImGui::PushStyleColor( ImGuiCol_Text, notificationParams[int( notification.type )].second );    
            ImGui::TextWrapped( "%s", notification.header.c_str() );
            if ( changeHeaderColor )
                ImGui::PopStyleColor();

            if ( boldFont )
                ImGui::PopFont();
        }

        if ( !notification.text.empty() )
        {
            ImGui::SetCursorPosX( contentShift.x );
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + contentShift.y );
            UI::transparentTextWrapped( "%s", notification.text.c_str() );
        }

        if ( notification.onButtonClick )
        {
            UI::TestEngine::pushTree( "Notification" + std::to_string( i ) );
            if ( notification.header.empty() && notification.text.empty() )
                ImGui::SetCursorPosX( contentShift.x );
            if ( UI::buttonCommonSize( notification.buttonName.c_str(), { buttonWidth, 0 } ) )
                notification.onButtonClick();
            UI::TestEngine::popTree();
        }

        bool isHovered = false;
        if ( activeModal )
        {
            // workaround to be able to hover notification even if modal is present
            auto mousePos = ImGui::GetMousePos();
            isHovered = window->Rect().Contains( mousePos ) && !activeModal->Rect().Contains( mousePos );
        }
        else
        {
            isHovered = ImGui::IsWindowHovered();
        }
        if ( !isHovered )
            timer += ImGui::GetIO().DeltaTime;

        if ( isHovered )
        {
            drawList->PushClipRectFullScreen();
            const ImU32 color = isHovered ? ImGui::GetColorU32( ImGuiCol_Text ) : notificationParams[int( notification.type )].second;
            drawList->AddRect( window->Rect().Min, window->Rect().Max, color, 4.0f * scaling, 0, 2.0f * scaling );
            drawList->PopClipRect();
        }

        if ( !activeModal )
        {
            auto iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
            if ( iconsFont )
            {
                iconsFont->Scale = 0.7f;
                ImGui::PushFont( iconsFont );
            }
            ImGui::PushStyleColor( ImGuiCol_Border, 0 );
            ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetColorU32( ImGuiCol_WindowBg ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetColorU32( ImGuiCol_WindowBg ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetColorU32( ImGuiCol_WindowBg ) );
            ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
            std::string crossText = "\xef\x80\x8d";
            auto textWidth = ImGui::CalcTextSize( crossText.c_str() ).x;
            auto windRect = window->Rect();
            ImRect rect;
            ImVec2 size = ImVec2( textWidth + ImGui::GetStyle().FramePadding.x * 2, ImGui::GetFrameHeight() );
            rect.Min.x = windRect.Max.x - size.x;
            rect.Min.y = windRect.Min.y;
            auto cursorPos = ImGui::GetCursorPos();
            ImGui::SetCursorPos( ImVec2( width - size.x - cWindowBorderWidth * scaling, cWindowBorderWidth * scaling ) );
            if ( ImGui::Button( crossText.c_str(), size ) )
                filteredNumber = i;
            ImGui::SetCursorPos( cursorPos );
            ImGui::Dummy( ImVec2( 0, 0 ) );
            ImGui::PopStyleColor( 5 );
            if ( iconsFont )
            {
                iconsFont->Scale = 1.0f;
                ImGui::PopFont();
            }
        }

        if ( counter > 1 )
        {
            const ImU32 color = isHovered ? ImGui::GetColorU32( ImGuiCol_Text ) : notificationParams[int( notification.type )].second;
            const ImU32 textColor = ImGui::GetColorU32( ImGuiCol_WindowBg );
            auto countText = std::to_string( counter );
            auto textWidth = ImGui::CalcTextSize( countText.c_str() ).x;
            auto windRect = window->Rect();
            ImRect rect;
            ImVec2 size = ImVec2( textWidth + ImGui::GetStyle().FramePadding.x * 2, ImGui::GetFrameHeight() );
            rect.Min.x = windRect.Max.x;
            rect.Min.y = windRect.Max.y;
            rect.Min -= size * 0.5f;
            rect.Max = rect.Min + size;
            drawList->PushClipRectFullScreen();
            drawList->AddRectFilled( rect.Min, rect.Max, color, 4.0f * scaling, 0 );
            drawList->AddText( rect.Min + ImGui::GetStyle().FramePadding, textColor, countText.c_str() );
            drawList->PopClipRect();
        }

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar( 4 );
        currentPos.y -= window->Size.y;
    }
    if ( filteredNumber  > -1 )
        filterInvalid_( filteredNumber );
}

void RibbonNotifier::addNotification_( std::vector<NotificationWithTimer>& store, const RibbonNotification& notification )
{
    if ( !store.empty() && store.front().notification == notification )
    {
        store.front().sameCounter++;
        store.front().timer = 0.0f;
    }
    else
    {
        if ( store.size() == cNotificationNumberLimit )
            store.erase( store.end() - 1 );
        store.insert( store.begin(), NotificationWithTimer{ notification } );
    }
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
    if ( changed || ( !notifications_.empty() && !requestRedraw_ ) )
    {
#ifndef __EMSCRIPTEN__
        requestedTime_ = Time::max();
#endif
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
    requestRedraw_ = true;
#ifndef __EMSCRIPTEN__
    Time neededTime = std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( minTimeReq * 1000 ) + 100 );
    if ( requestedTime_ < neededTime )
        return;
    requestedTime_ = neededTime;
    asyncRequest_.request( requestedTime_, [&] ()
    {
        CommandLoop::appendCommand( [&] ()
        {
            getViewerInstance().incrementForceRedrawFrames();
            requestedTime_ = Time::max();
            requestRedraw_ = false;
        } );
    } );
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( postEmptyEvent( $0, 2 ), int( minTimeReq * 1000 ) + 100 );
#pragma clang diagnostic pop
#endif
}

bool RibbonNotification::operator==( const RibbonNotification& other ) const
{
    return
        header == other.header &&
        text == other.text &&
        buttonName == other.buttonName &&
        type == other.type &&
        !onButtonClick  &&
        !other.onButtonClick;
}

}