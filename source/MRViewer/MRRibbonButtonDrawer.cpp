#include "MRRibbonButtonDrawer.h"
#include "MRRibbonMenu.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRImGuiImage.h"
#include "MRShortcutManager.h"
#include "ImGuiHelpers.h"
#include "MRRibbonIcons.h"
#include "imgui_internal.h"

namespace MR
{

void RibbonButtonDrawer::InitGradientTexture()
{
    auto& texture = GetGradientTexture();
    if ( !texture )
        texture = std::make_unique<ImGuiImage>();
    MeshTexture data;
    data.resolution = Vector2i( 1, 2 );
    data.pixels = {
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientEnd )
    };
    data.filter = MeshTexture::FilterType::Linear;
    texture->update( data );
}

std::unique_ptr<ImGuiImage>& RibbonButtonDrawer::GetGradientTexture()
{
    static std::unique_ptr<ImGuiImage> texture;
    return texture;
}

bool RibbonButtonDrawer::GradientButton( const char* label, const ImVec2& size /*= ImVec2( 0, 0 ) */ )
{
    auto& texture = GetGradientTexture();
    if ( !texture )
        return ImGui::Button( label, size );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1, 1, 1, 1 ) );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );

    int pushedStyleNum = 1;
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    if ( size.y == 0 )
    {
        auto framePadding = style.FramePadding;
        framePadding.y = cGradientButtonFramePadding;
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            framePadding.y *= menu->menu_scaling();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
        ++pushedStyleNum;
    }

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 realSize = ImGui::CalcItemSize( size, labelSize.x + style.FramePadding.x * 2.0f, labelSize.y + style.FramePadding.y * 2.0f );
    const ImRect bb( pos, ImVec2( pos.x + realSize.x, pos.y + realSize.y ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0, 0 ), ImVec2( 1, 1 ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::Button( label, size );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientCheckbox( const char* label, bool* value )
{
    auto& texture = GetGradientTexture();
    if ( !texture || ( value && !*value ) )
        return ImGui::Checkbox( label, value );

    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_CheckMark, ImVec4( 1, 1, 1, 1 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const float clickSize = ImGui::GetFrameHeight();

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0, 0 ), ImVec2( 1, 1 ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::Checkbox( label, value );

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientRadioButton( const char* label, int* value, int v_button )
{
    auto& texture = GetGradientTexture();
    if ( !texture || ( value && ( *value != v_button ) ) )
        return ImGui::RadioButton( label, value, v_button );

    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_CheckMark, ImVec4( 1, 1, 1, 1 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const float clickSize = ImGui::GetFrameHeight();

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0, 0 ), ImVec2( 1, 1 ),
        Color::white().getUInt32(), clickSize * 0.5f );

    auto res = ImGui::RadioButton( label, value, v_button );

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 2 );
    return res;
}

RibbonButtonDrawer::ButtonItemWidth RibbonButtonDrawer::calcItemWidth( const MenuItemInfo& item, DrawButtonParams::SizeType sizeType )
{
    ButtonItemWidth res;
    if ( sizeType == DrawButtonParams::SizeType::Big )
    {
        const float cMinItemSize = cRibbonItemMinWidth * scaling_;

        float maxTextWidth = 0.f;
        for ( const auto& i : item.captionSize.splitInfo )
            maxTextWidth = std::max( maxTextWidth, i.second );

        res.baseWidth = maxTextWidth + 2 * cRibbonButtonWindowPaddingX * scaling_;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
        {
            auto additionalSize = 3 * cSmallIconSize * scaling_;
            if ( cMinItemSize - res.baseWidth < additionalSize )
                res.baseWidth += additionalSize;
        }

        if ( res.baseWidth < cMinItemSize )
            res.baseWidth = cMinItemSize;
        return res;
    }
    else if ( sizeType == DrawButtonParams::SizeType::SmallText )
    {
        res.baseWidth = ( cSmallIconSize +
                          2.0f * cRibbonButtonWindowPaddingX +
                          2.0f * cRibbonItemInterval ) * scaling_ +
            item.captionSize.baseSize;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
            res.additionalWidth = cSmallItemDropSizeModifier * ( cSmallIconSize + 2.0f * cRibbonButtonWindowPaddingX ) * scaling_;
    }
    else
    {
        res.baseWidth = ( cSmallIconSize + 2.0f * cRibbonButtonWindowPaddingX ) * scaling_;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
            res.additionalWidth = cSmallItemDropSizeModifier * res.baseWidth;
    }
    return res;
}

void RibbonButtonDrawer::drawButtonItem( const MenuItemInfo& item, const DrawButtonParams& params )
{
    std::string requirements = getRequirements_( item.item );

    bool dropItem = item.item->type() == RibbonItemType::ButtonWithDrop;

    ImVec2 itemSize = params.itemSize;
    if ( dropItem && params.sizeType == DrawButtonParams::SizeType::Small )
        itemSize.x += params.itemSize.x * cSmallItemDropSizeModifier;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    ImGui::BeginChild( ( "##childGroup" + item.item->name() ).c_str(), itemSize, false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse );
    ImGui::PopStyleVar();

    ImGui::BeginGroup();

    int colorChanged = pushRibbonButtonColors_( requirements.empty(), item.item->isActive(), params.rootType );
    bool pressed = ImGui::Button( ( "##wholeChildBtn" + item.item->name() ).c_str(), itemSize );

    ImFont* font = nullptr;
    if ( fontManager_ ) {
        auto& fontManager = *fontManager_;
        font = fontManager_->getFontByType( RibbonFontManager::FontType::Icons );
        if ( params.iconSize != 0 )
            font->Scale = params.iconSize / fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        else if ( params.sizeType != DrawButtonParams::SizeType::Big )
            font->Scale = cSmallIconSize / fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * font->Scale * scaling_ );
    ImVec2 iconRealSize = ImVec2( imageRequiredSize, imageRequiredSize );
    bool needWhiteIcon = !requirements.empty() || item.item->isActive() || params.rootType != DrawButtonParams::Ribbon;
    auto* imageIcon = RibbonIcons::findByName( item.item->name(), iconRealSize.x, needWhiteIcon ?
                                               RibbonIcons::ColorType::White : RibbonIcons::ColorType::Colored,
                                               RibbonIcons::IconType::RibbonItemIcon );

    if ( !imageIcon )
        iconRealSize = ImGui::CalcTextSize( item.icon.c_str() );

    if ( params.sizeType != DrawButtonParams::SizeType::SmallText )
    {
        ImGui::SetCursorPosX( ( params.itemSize.x - iconRealSize.x ) * 0.5f );
    }
    else
    {
        if ( !imageIcon )
            ImGui::SetCursorPosX( ImGui::GetStyle().WindowPadding.x );
        else
            ImGui::SetCursorPosX( ImGui::GetStyle().WindowPadding.x * 0.5f );
    }

    if ( params.sizeType != DrawButtonParams::SizeType::Big )
        ImGui::SetCursorPosY( ( params.itemSize.y - iconRealSize.y ) * 0.5f );
    else
    {
        if ( !imageIcon )
            ImGui::SetCursorPosY( 2.0f * ImGui::GetStyle().WindowPadding.y );
        else
            ImGui::SetCursorPosY( ImGui::GetStyle().WindowPadding.y );
    }

    if ( !imageIcon )
        ImGui::Text( "%s", item.icon.c_str() );
    else
    {
        ImVec4 multColor = ImVec4( 1, 1, 1, 1 );
        if ( needWhiteIcon )
            multColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        ImGui::Image( *imageIcon, iconRealSize, multColor );
    }

    if ( fontManager_ )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    if ( params.sizeType == DrawButtonParams::SizeType::Big )
    {
        const float availableHeight = ImGui::GetContentRegionAvail().y;
        const int numLines = int( item.captionSize.splitInfo.size() );
        const float textHeight = numLines * ImGui::GetTextLineHeight() + ( numLines - 1 ) * ImGui::GetStyle().ItemSpacing.y;

        if ( !imageIcon )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f );
        else
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f - ImGui::GetStyle().WindowPadding.y );

        for ( const auto& i : item.captionSize.splitInfo )
        {
            ImGui::SetCursorPosX( ( params.itemSize.x - i.second ) * 0.5f );
            ImGui::TextUnformatted( &i.first.front(), &i.first.back() + 1 );
        }
    }
    else if ( params.sizeType == DrawButtonParams::SizeType::SmallText )
    {
        ImGui::SameLine();
        ImGui::SetCursorPosY( ( params.itemSize.y - ImGui::GetTextLineHeight() ) * 0.5f );
        const auto& caption = item.caption.empty() ? item.item->name() : item.caption;
        ImGui::Text( "%s", caption.c_str() );
    }

    if ( colorChanged > 0 )
        ImGui::PopStyleColor( colorChanged );

    ImGui::EndGroup();

    if ( pressed )
        onPressAction_( item.item, requirements.empty() );

    if ( ImGui::IsItemHovered() )
        drawTooltip_( item, requirements );

    if ( dropItem )
        drawButtonDropItem_( item, params, requirements.empty() );
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

bool RibbonButtonDrawer::drawCustomStyledButton( const char* icon, const ImVec2& size, float iconSize )
{
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrab ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    ImFont* font = nullptr;
    if ( fontManager_ )
    {
        font = fontManager_->getFontByType( RibbonFontManager::FontType::Icons );
        font->Scale = iconSize / fontManager_->getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );
    }

    bool pressed = ImGui::Button( icon, size );

    if ( fontManager_ )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar();
    return pressed;
}

void RibbonButtonDrawer::drawButtonDropItem_( const MenuItemInfo& item, const DrawButtonParams& params, bool enabled )
{
    float iconSize = params.iconSize * 0.5f;
    ImVec2 itemSize = ImVec2( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );
    ImVec2 dropBtnPos;
    if ( params.sizeType == DrawButtonParams::SizeType::Small )
    {
        itemSize.x = params.itemSize.x * cSmallItemDropSizeModifier;
        itemSize.y = params.itemSize.y;
        dropBtnPos.x = params.itemSize.x;
        dropBtnPos.y = 0.0f;
    }
    else if ( params.sizeType == DrawButtonParams::SizeType::SmallText )
    {
        itemSize.x = params.itemSize.y;
        itemSize.y = params.itemSize.y;
        dropBtnPos.x = params.itemSize.x - itemSize.x;
        dropBtnPos.y = 0.0f;
    }
    else
    {
        assert( params.sizeType == DrawButtonParams::SizeType::Big );
        dropBtnPos.x = params.itemSize.x - itemSize.x;
        dropBtnPos.y = params.itemSize.y - itemSize.y;
    }
    ImGui::SetCursorPos( dropBtnPos );
    auto absMinPos = ImGui::GetCurrentContext()->CurrentWindow->DC.CursorPos;

    auto name = "##DropDown" + item.item->name();
    auto nameWindow = name + "Popup";
    bool menuOpened = ImGui::IsPopupOpen( nameWindow.c_str() );
    ImGui::SetItemAllowOverlap();

    bool dropBtnEnabled = enabled && !item.item->dropItems().empty();

    int pushedColors = pushRibbonButtonColors_( enabled, menuOpened, params.rootType );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    bool comboPressed = ImGui::Button( name.c_str(), itemSize ) && dropBtnEnabled;

    ImFont* font = nullptr;
    if ( fontManager_ )
    {
        auto& fontManager = *fontManager_;
        font = fontManager.getFontByType( RibbonFontManager::FontType::Icons );
        if ( params.sizeType == DrawButtonParams::SizeType::Big )
            font->Scale = iconSize / fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        else
            font->Scale = iconSize*1.5f / fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );
    }
    auto iconRealSize = ImGui::CalcTextSize( "\xef\x81\xb8" ); //down icon
    ImGui::SetCursorPosX( dropBtnPos.x + ( itemSize.x - iconRealSize.x + 1 ) * 0.5f );
    ImGui::SetCursorPosY( dropBtnPos.y + ( itemSize.y - iconRealSize.y - 1 ) * 0.5f );
    ImGui::Text( "%s", "\xef\x81\xb8" );
    
    ImGui::PopStyleVar();
    ImGui::PopStyleColor( pushedColors );

    if ( fontManager_ )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    // manage menu popup
    if ( comboPressed && !menuOpened )
        ImGui::OpenPopup( nameWindow.c_str() );

    if ( !menuOpened )
        return;

    if ( ImGuiWindow* menuWindow = ImGui::FindWindowByName( nameWindow.c_str() ) )
        if ( menuWindow->WasActive )
        {
            ImRect frame;
            frame.Min = absMinPos;
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetWindowAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    ImGui::Begin( nameWindow.c_str(), NULL, window_flags );
    if ( menuOpened )
    {
        drawDropList_( item.item );
        ImGui::EndPopup();
    }
}

void RibbonButtonDrawer::drawDropList_( const std::shared_ptr<RibbonMenuItem>& baseDropItem )
{
    auto dropList = baseDropItem->dropItems();
    if ( dropList.empty() )
        ImGui::CloseCurrentPopup();
    for ( int i = 0; i < dropList.size(); ++i )
    {
        const auto& dropItem = dropList[i];
        bool itemWithDrop = dropItem->type() == RibbonItemType::ButtonWithDrop;

        if ( i > 0 && itemWithDrop )
            ImGui::Separator();

        std::string requirements = getRequirements_( dropItem );

        std::string caption = dropItem->name();

        const auto& schema = RibbonSchemaHolder::schema();
        auto it = schema.items.find( dropItem->name() );
        if ( it != schema.items.end() && it->second.caption != "" )
            caption = it->second.caption;
        auto pressed = ImGui::MenuItem(
            ( caption + "##dropItem" ).c_str(),
            nullptr,
            dropItem->isActive(),
            requirements.empty() );

        if ( pressed )
            onPressAction_( dropItem, requirements.empty() );

        if ( ImGui::IsItemHovered() && menu_ )
        {
            if ( it != schema.items.end() )
                drawTooltip_( it->second, requirements );
        }

        if ( itemWithDrop )
        {
            if ( ImGui::BeginMenu( ( "More...##recursiveDropMenu" + dropItem->name() ).c_str(), requirements.empty() ) )
            {
                drawDropList_( dropItem );
                ImGui::EndMenu();
            }
            if ( i + 1 < dropList.size() )
                ImGui::Separator();
        }
    }
}

void RibbonButtonDrawer::drawTooltip_( const MenuItemInfo& item, const std::string& requirements )
{
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 0 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cRibbonButtonWindowPaddingX * scaling_, cRibbonButtonWindowPaddingY * scaling_ ) );
    const std::string& tooltip = item.tooltip;

    const auto& caption = item.caption.empty() ? item.item->name() : item.caption;

    std::string fullText;
    fullText = caption;
    std::string shortcutStr;

    if ( shortcutManager_ )
    {
        auto shortcut = shortcutManager_->findShortcutByName( item.item->name() );
        if ( shortcut )
        {
            shortcutStr = " (" + ShortcutManager::getKeyString( *shortcut ) + ")";
            fullText += shortcutStr;
        }
    }

    if ( !tooltip.empty() )
    {
        fullText += '\n';
        fullText += tooltip;
    }
    if ( !requirements.empty() )
    {
        fullText += '\n';
        fullText += requirements;
    }
    auto textSize = ImGui::CalcTextSize( fullText.c_str(), NULL, false, 400.f );

    ImGui::SetNextWindowContentSize( textSize );
    ImGui::BeginTooltip();
    ImGui::Text( "%s%s", caption.c_str(), shortcutStr.c_str() );
    if ( !tooltip.empty() )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) );
        ImGui::TextWrapped( "%s", tooltip.c_str() );
        ImGui::PopStyleColor();
    }
    if ( !requirements.empty() )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
        ImGui::TextWrapped( "%s", requirements.c_str() );
        ImGui::PopStyleColor();
    }
    ImGui::EndTooltip();
    ImGui::PopStyleVar( 2 );
}


int RibbonButtonDrawer::pushRibbonButtonColors_( bool enabled, bool active, DrawButtonParams::RootType rootType ) const
{
    if ( active )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveClicked ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActive ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextEnabled ).getUInt32() );
        return 4;
    }
    if ( !enabled )
    {
        if ( rootType == DrawButtonParams::RootType::Header )
        {
            auto tabTextColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText );
            tabTextColor.a = 77;
            ImGui::PushStyleColor( ImGuiCol_Text, tabTextColor.getUInt32() );
        }
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
    }
    else
    {
        if ( rootType == DrawButtonParams::RootType::Header )
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Text ).getUInt32() );
    }

    ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
    if ( rootType == DrawButtonParams::RootType::Ribbon )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonClicked ).getUInt32() );
    }
    else if ( rootType == DrawButtonParams::RootType::Toolbar )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
    }
    else 
    {
        assert( rootType == DrawButtonParams::RootType::Header );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabClicked ).getUInt32() );
    }
    return 4;
}

}
