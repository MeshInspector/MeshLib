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
#include "MRMouseController.h"
#include <imgui_internal.h>

namespace
{
constexpr int cNotificationNumberLimit = 10;

}

namespace MR
{

void RibbonNotifier::pushNotification( const RibbonNotification& notification )
{
    if ( !historyMode_ && bool( notification.tags & allowedTagMask ) )
        addNotification_( notifications_, notification );
    addNotification_( notificationsHistory_, notification );

    currentHistoryBtnTimer_ = showHistoryBtnMaxTime_;
    scrollDownNeeded_ = true;    
    requestClosestRedraw_();
}

void RibbonNotifier::draw( float scaling, const Box2i& limitFramebuffer )
{
    drawHistoryButton_( scaling, limitFramebuffer );
    if ( historyMode_ )
        drawHistory_( scaling, limitFramebuffer );
    else
        drawFloating_( scaling, limitFramebuffer );
    filterInvalid_( -1 );
}

void RibbonNotifier::setHitoryButtonMaxLifeTime( float histBtnMaxLifeTime )
{
    if ( showHistoryBtnMaxTime_ == histBtnMaxLifeTime )
        return; // do nothing
    if ( showHistoryBtnMaxTime_ <= 0 && histBtnMaxLifeTime <= 0 )
        return; // do nothing

    if ( showHistoryBtnMaxTime_ <= 0 )
    {
        currentHistoryBtnTimer_ = histBtnMaxLifeTime;
    }
    else
    {
        if ( currentHistoryBtnTimer_ > 0 )
            currentHistoryBtnTimer_ += ( histBtnMaxLifeTime - showHistoryBtnMaxTime_ ); // decrease current timer
    }
    showHistoryBtnMaxTime_ = histBtnMaxLifeTime;
    requestClosestRedraw_();
}

void RibbonNotifier::drawHistoryButton_( float scaling, const Box2i& limitFramebuffer )
{
    using namespace StyleConsts::Notification;
    if ( notificationsHistory_.empty() )
        return;

    if ( showHistoryBtnMaxTime_ > 0 )
    {
        if ( currentHistoryBtnTimer_ >= 0 && !historyMode_ )
            currentHistoryBtnTimer_ -= ImGui::GetIO().DeltaTime;
        if ( currentHistoryBtnTimer_ < 0 )
            return;
    }

    const auto windowSzie = ImVec2( 36, cHistoryButtonSizeY ) * scaling;
    Vector2f windowPos = Vector2f( float( limitFramebuffer.min.x ), float( getViewerInstance().framebufferSize.y - limitFramebuffer.min.y ) - windowSzie.y );
    if ( cornerPosition == RibbonNotificationCorner::LowerRight )
        windowPos.x = float( limitFramebuffer.max.x ) - windowSzie.x;

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSzie, ImGuiCond_Always );
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoBringToFrontOnFocus;
    std::string name = "##NotificationButton";
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    auto windowBgColor = MR::ColorTheme::getViewportColor( MR::ColorTheme::ViewportColorsType::Borders );
    if ( ColorTheme::getPreset() == ColorTheme::Preset::Dark )
        windowBgColor = windowBgColor.scaledAlpha( 0.5f );
    ImGui::PushStyleColor( ImGuiCol_WindowBg, windowBgColor.scaledAlpha( 0.6f ).getUInt32() );

    ImGui::Begin( name.c_str(), nullptr, flags );

    auto iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( iconsFont )
    {
        iconsFont->Scale = 0.65f;
        ImGui::PushFont( iconsFont );
    }

    auto fontSize = ImGui::GetFontSize();
    ImGui::SetCursorPos( 0.5f * ( windowSzie - ImVec2( fontSize, fontSize ) ) );
    ImGui::PushStyleColor( ImGuiCol_Text, UI::notificationChar( notificationsHistory_.front().notification.type ).second );
    ImGui::Text( "%s", UI::notificationChar( notificationsHistory_.front().notification.type ).first );
    ImGui::PopStyleColor();

    if ( iconsFont )
    {
        iconsFont->Scale = 1.0f;
        ImGui::PopFont();
    }

    if ( ImGui::IsWindowHovered() )
    {
        auto window = ImGui::GetCurrentContext()->CurrentWindow;
        if ( ImGui::IsMouseClicked( ImGuiMouseButton_Left ) )
        {
            historyMode_ = !historyMode_;
            if ( historyMode_ )
            {
                notifications_.clear();
                scrollDownNeeded_ = true;
            }
            else
            {
                currentHistoryBtnTimer_ = showHistoryBtnMaxTime_;
                if ( currentHistoryBtnTimer_ > 0 )
                    requestClosestRedraw_();
            }
        }

        auto drawList = window->DrawList;
        drawList->PushClipRectFullScreen();
        const ImU32 color = ImGui::GetColorU32( ImGuiCol_Text );
        drawList->AddRect( window->Rect().Min, window->Rect().Max, color, cWindowRounding * scaling, 0, cWindowBorderWidth * scaling );
        drawList->PopClipRect();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 3 );
}

void RibbonNotifier::drawHistory_( float scaling, const Box2i& limitFramebuffer )
{
    using namespace StyleConsts::Notification;
    const float width = 351.0f * scaling;
    Vector2f windowPos = Vector2f( float( limitFramebuffer.min.x ), float( getViewerInstance().framebufferSize.y - limitFramebuffer.min.y ) - cHistoryButtonSizeY * scaling );
    if ( cornerPosition == RibbonNotificationCorner::LowerRight )
        windowPos.x = float( limitFramebuffer.max.x ) - width;

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always, ImVec2( 0.f, 1.0f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( width, 1 ), ImVec2( width, float( limitFramebuffer.max.y - limitFramebuffer.min.y ) - cHistoryButtonSizeY * scaling ) );
    ImGui::SetNextWindowSize( ImVec2( width, -1 ), ImGuiCond_Always );
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoMove;
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cNotificationWindowPaddingX * scaling, cNotificationWindowPaddingY * scaling ) );
    ImGui::PushStyleColor( ImGuiCol_WindowBg, MR::ColorTheme::getViewportColor( MR::ColorTheme::ViewportColorsType::Borders ).scaledAlpha( 0.4f ).getUInt32() );
    ImGui::Begin( "NotificationsHistory", nullptr, flags );

    const float padding = cWindowPadding * scaling;
    Vector2f currentPos = Vector2f( windowPos.x + padding, windowPos.y );
    const float notWidth = 319.0f * scaling;
    for ( int i = 0; i < notificationsHistory_.size(); ++i )
    {
        currentPos.y -= padding;
        drawNotification_( { .index = i,.scalig = scaling,.width = notWidth,.historyMode = true,.currentPos = &currentPos } );
    }

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    if ( scrollDownNeeded_ || prevHistoryScrollMax_ != window->ScrollMax.y )
    {
        
        window->ScrollTargetCenterRatio.y = 0.0f;
        window->ScrollTarget.y = window->ScrollMax.y;
        scrollDownNeeded_ = false;
        prevHistoryScrollMax_ = window->ScrollMax.y;
    }

    auto lostFoucsClickCheck = [] ()->bool
    {
        if ( ImGui::IsWindowAppearing() )
            return false;
        if ( ImGui::IsWindowHovered( ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem ) )
            return false;
        if ( ImGui::IsMouseClicked( ImGuiMouseButton_Left ) || ImGui::IsMouseClicked( ImGuiMouseButton_Right ) || ImGui::IsMouseClicked( ImGuiMouseButton_Right ) )
            return true;
        if ( ImGui::GetIO().WantCaptureMouse )
            return false;
        const auto& mCtrl = getViewerInstance().mouseController();
        if ( mCtrl.isPressed( MouseButton::Left ) || mCtrl.isPressed( MouseButton::Right ) || mCtrl.isPressed( MouseButton::Middle ) )
            return true;
        return false;
    };

    if ( lostFoucsClickCheck() )
    {
        historyMode_ = false;
        currentHistoryBtnTimer_ = showHistoryBtnMaxTime_;
        if ( currentHistoryBtnTimer_ > 0 )
            requestClosestRedraw_();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 3 );
}

void RibbonNotifier::drawFloating_( float scaling, const Box2i& limitFramebuffer )
{
    using namespace StyleConsts::Notification;
    const float padding = cWindowPadding * scaling;
    const float width = 319.0f * scaling;
    Vector2f currentPos = Vector2f( float( limitFramebuffer.min.x ) + padding, float( getViewerInstance().framebufferSize.y - limitFramebuffer.min.y ) - cHistoryButtonSizeY * scaling );
    if ( cornerPosition == RibbonNotificationCorner::LowerRight )
        currentPos.x = float( limitFramebuffer.max.x ) - padding - width;

    int filteredNumber = -1;
    for ( int i = 0; i < notifications_.size(); ++i )
    {
        currentPos.y -= padding;
        if ( !drawNotification_( { .index = i,.scalig = scaling,.width = width,.historyMode = false,.currentPos = &currentPos } ) )
            filteredNumber = i;
    }
    if ( filteredNumber > -1 )
        filterInvalid_( filteredNumber );
}

bool RibbonNotifier::drawNotification_( const DrawNotificationSettings& settings )
{
    using namespace StyleConsts::Notification;
    auto& [notification, timer, counter] = settings.historyMode ? notificationsHistory_[settings.index] : notifications_[settings.index];

    assert( settings.currentPos );
    if ( !settings.currentPos )
        return false;

    const auto scaling = settings.scalig;

    ImGuiWindow* parentWindow{ nullptr };
    if ( settings.historyMode )
        parentWindow = ImGui::GetCurrentContext()->CurrentWindow;

    if ( settings.historyMode )
    {
        // hack to correct calculate size of parent window
        //ImGui::Button( "1", ImVec2( settings.width, 1 ) ); // DEBUG line, might be useful to change correction if needed
        ImGui::Dummy( ImVec2( settings.width, 1 ) );
    }

    auto windowPos = *settings.currentPos;
    if ( settings.historyMode )
        windowPos.y += ( parentWindow->ScrollMax.y - parentWindow->Scroll.y );

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always, ImVec2( 0.f, 1.0f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( settings.width, 1 ), ImVec2( settings.width, settings.width ) );
    ImGui::SetNextWindowSize( ImVec2( settings.width, -1 ), ImGuiCond_Always );

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoTitleBar | ( settings.historyMode ? ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_AlwaysUseWindowPadding : 0 ) |
        ImGuiWindowFlags_NoMove;
    std::string name = "##notification" + std::to_string( settings.index );
    ImGui::PushStyleVar( settings.historyMode ? ImGuiStyleVar_ChildBorderSize : ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushStyleVar( settings.historyMode ? ImGuiStyleVar_ChildRounding : ImGuiStyleVar_WindowRounding, cWindowRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cNotificationWindowPaddingX * scaling, cNotificationWindowPaddingY * scaling ) );
    auto windowBgColor = MR::ColorTheme::getViewportColor( MR::ColorTheme::ViewportColorsType::Borders );
    if ( ColorTheme::getPreset() == ColorTheme::Preset::Dark )
        windowBgColor = windowBgColor.scaledAlpha( 0.6f );
    ImGui::PushStyleColor( settings.historyMode ? ImGuiCol_ChildBg : ImGuiCol_WindowBg, windowBgColor.getUInt32() );

    auto activeModal = settings.historyMode ? nullptr : ImGui::GetTopMostPopupModal();

    const float closeBtnSize = 16 * scaling;
    const float closeBtnPadding = 12 * scaling;
    const bool hasCloseBtn = !settings.historyMode && !activeModal;
    const bool hasCounter = counter > 1;

    if ( !settings.historyMode && settings.index + 1 == cNotificationNumberLimit )
        ImGui::SetNextWindowBgAlpha( 0.5f );
    ImGui::Begin( name.c_str(), nullptr, flags );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;

    if ( !settings.historyMode && ImGui::IsWindowAppearing() )
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
    cirlcePos.y += window->Scroll.y;
    drawList->AddCircleFilled( cirlcePos, 3.0f * scaling, UI::notificationChar( notification.type ).second );

    const bool changeHeaderColor = notification.type == NotificationType::Error || notification.type == NotificationType::Warning;

    const ImVec2 contentShift = ImVec2( 26.0f * scaling, radius );
    auto boldFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::SemiBold );
    if ( !notification.header.empty() )
    {
        if ( boldFont )
            ImGui::PushFont( boldFont );

        ImGui::SetCursorPosX( contentShift.x );
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + contentShift.y );
        if ( changeHeaderColor )
            ImGui::PushStyleColor( ImGuiCol_Text, UI::notificationChar( notification.type ).second );

        const auto backupWorkRect = window->WorkRect.Max;
        if ( hasCloseBtn || hasCounter )
            window->WorkRect.Max.x -= ( closeBtnSize + closeBtnPadding );
        ImGui::TextWrapped( "%s", notification.header.c_str() );
        window->WorkRect.Max = backupWorkRect;

        if ( changeHeaderColor )
            ImGui::PopStyleColor();

        if ( boldFont )
            ImGui::PopFont();
    }

    if ( !notification.text.empty() )
    {
        ImGui::SetCursorPosX( contentShift.x );
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + contentShift.y );
        const auto backupWorkRect = window->WorkRect.Max;
        if ( hasCloseBtn || hasCounter )
            window->WorkRect.Max.x -= ( closeBtnSize + closeBtnPadding );
        UI::transparentTextWrapped( "%s", notification.text.c_str() );
        window->WorkRect.Max = backupWorkRect;
    }

    if ( notification.onButtonClick )
    {
        UI::TestEngine::pushTree( "Notification" + std::to_string( settings.index ) );
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
        auto windRect = window->Rect();
        if ( settings.historyMode )
            drawList->PushClipRect( parentWindow->InnerClipRect.Min, parentWindow->InnerClipRect.Max );
        else
            drawList->PushClipRectFullScreen();
        const ImU32 color = isHovered ? ImGui::GetColorU32( ImGuiCol_Text ) : UI::notificationChar( notification.type ).second;
        drawList->AddRect( windRect.Min, windRect.Max, color, 4.0f * scaling, 0, 2.0f * scaling );
        drawList->PopClipRect();
    }

    bool returnValue = true;

    if ( hasCloseBtn )
    {
        if ( settings.historyMode )
            drawList->PushClipRect( parentWindow->InnerClipRect.Min, parentWindow->InnerClipRect.Max );
        else
            drawList->PushClipRectFullScreen();

        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
        ImGui::PushStyleColor( ImGuiCol_Button, 0 );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, 0 );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.5, 0.5, 0.5, 0.5 ) );
        ImGui::SetCursorPos( ImVec2( settings.width - closeBtnPadding - closeBtnSize, closeBtnPadding ) + window->Scroll );
        auto screenPos = ImGui::GetCursorScreenPos();
        if ( ImGui::Button( "##closeNotification", ImVec2( closeBtnSize, closeBtnSize ) ) )
            returnValue = false;
        const float innerCrossPadding = 4 * scaling;
        drawList->AddLine(
            ImVec2( screenPos.x + innerCrossPadding - 1, screenPos.y + innerCrossPadding - 1 ),
            ImVec2( screenPos.x + closeBtnSize - innerCrossPadding, screenPos.y + closeBtnSize - innerCrossPadding ),
            Color::gray().getUInt32(), scaling );
        drawList->AddLine(
            ImVec2( screenPos.x + closeBtnSize - innerCrossPadding, screenPos.y + innerCrossPadding - 1 ),
            ImVec2( screenPos.x + innerCrossPadding - 1, screenPos.y + closeBtnSize - innerCrossPadding ),
            Color::gray().getUInt32(), scaling );
        ImGui::PopStyleColor( 3 );
        ImGui::PopStyleVar();
        drawList->PopClipRect();
    }

    if ( hasCounter )
    {
        if ( boldFont )
            ImGui::PushFont( boldFont );
        auto countText = std::to_string( counter );
        const auto textSize = ImGui::CalcTextSize( countText.c_str() );

        auto windRect = window->Rect();
        const float minCounterPosY = hasCloseBtn ? windRect.Min.y + closeBtnPadding + closeBtnSize + 6 * scaling : windRect.Min.y + closeBtnPadding;

        const auto counterRadius = 12 * scaling;
        auto counterPos = windRect.Max - ImVec2( closeBtnPadding + closeBtnSize, closeBtnPadding + closeBtnSize );
        if ( counterPos.y < minCounterPosY )
        {
            counterPos.y = minCounterPosY;
            ImGui::SetCursorScreenPos( counterPos );
            ImGui::Dummy( ImVec2( closeBtnSize, closeBtnSize ) );
        }
        const auto counterCenter = ImVec2( counterPos.x + closeBtnSize * 0.5f, counterPos.y + closeBtnSize * 0.5f );
        if ( settings.historyMode )
            drawList->PushClipRect( parentWindow->InnerClipRect.Min, parentWindow->InnerClipRect.Max );
        else
            drawList->PushClipRectFullScreen();
        drawList->AddCircleFilled( counterCenter, counterRadius, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ).getUInt32() );
        if ( changeHeaderColor )
            ImGui::PushStyleColor( ImGuiCol_Text, UI::notificationChar( notification.type ).second );
        drawList->AddCircle( counterCenter, counterRadius, ImGui::GetColorU32( ImGuiCol_Text ), 0, scaling );
        drawList->AddText( counterCenter - textSize * 0.5f, ImGui::GetColorU32( ImGuiCol_Text ), countText.c_str() );
        if ( changeHeaderColor )
            ImGui::PopStyleColor();
        drawList->PopClipRect();

        if ( boldFont )
            ImGui::PopFont();
    }

    if ( settings.historyMode )
        ImGui::EndChild();
    else
        ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 3 );

    if ( settings.historyMode )
    {
        // hack to correct calculate size of parent window
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - 1 - 3 * scaling ); // correction
        //ImGui::Button( "2", ImVec2( settings.width, 1 ) ); // DEBUG line, might be useful to change correction if needed
        ImGui::Dummy( ImVec2( settings.width, 1 ) );
    }

    settings.currentPos->y -= window->Size.y;
    return returnValue;
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
        if ( store.front().notification.lifeTimeSec < 0.0f )
            store.front().notification.lifeTimeSec = defaultNotificationLifeTimeSeconds;
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

    if ( showHistoryBtnMaxTime_ > 0 && currentHistoryBtnTimer_ > 0 )
    {
        if ( currentHistoryBtnTimer_ < minTimeReq )
            minTimeReq = currentHistoryBtnTimer_;
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