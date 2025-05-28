#include "MRRibbonButtonDrawer.h"
#include "MRRibbonMenu.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRImGuiImage.h"
#include "MRShortcutManager.h"
#include "ImGuiHelpers.h"
#include "MRRibbonIcons.h"
#include "MRViewer/MRUITestEngine.h"
#include "MRViewerInstance.h"
#include "MRUIStyle.h"
#include "imgui_internal.h"

namespace MR
{

float getScaling()
{
    const auto menu = ImGuiMenu::instance();
    if ( menu )
        return menu->menu_scaling();
    return 1.f;
}

std::vector<std::unique_ptr<MR::ImGuiImage>> RibbonButtonDrawer::textures_ = std::vector<std::unique_ptr<MR::ImGuiImage>>( int( RibbonButtonDrawer::TextureType::Count ) );

void DrawCustomArrow( ImDrawList* drawList, const ImVec2& startPoint, const ImVec2& midPoint, const ImVec2& endPoint, ImU32 col, float thickness )
{
    drawList->PathLineTo( startPoint );
    drawList->PathLineTo( midPoint );
    drawList->PathLineTo( endPoint );
    drawList->PathStroke( col, 0, thickness );

    const float radius = thickness * 0.5f;
    drawList->AddCircleFilled( startPoint, radius, col );
    drawList->AddCircleFilled( midPoint, radius, col );
    drawList->AddCircleFilled( endPoint, radius, col );
}

void RibbonButtonDrawer::InitGradientTexture()
{
    auto& textureM = GetTexture( TextureType::Mono );
    if ( !textureM )
        textureM = std::make_unique<ImGuiImage>();
    MeshTexture data;
    data.resolution = Vector2i( 1, 1 );
    data.pixels = { Color::white() };
    data.filter = FilterType::Linear;
    textureM->update( data );

    auto& textureG = GetTexture( TextureType::Gradient );
    if ( !textureG )
        textureG = std::make_unique<ImGuiImage>();
    data.resolution = Vector2i( 1, 2 );
    data.pixels = {
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientEnd )
    };
    data.filter = FilterType::Linear;
    textureG->update( data );


    auto& textureR = GetTexture( TextureType::RainbowRect );
    if ( !textureR )
        textureR = std::make_unique<ImGuiImage>();
    const int resX = 4;
    const int resY = 2;
    data.resolution = Vector2i( resX, resY );
    data.pixels.resize( resX * resY );
    float h, r, g, b;
    for ( int i = 0; i < resX; ++i )
    {
        h = ( 3.5f - 2.f * i / ( resX - 1.f ) ) / 6.f;
        ImGui::ColorConvertHSVtoRGB( h, 1.f, 1.f, r, g, b );
        data.pixels[i] = Color( r, g, b );

        h = ( 5.f + 2.f * i / ( resX - 1.f ) ) / 6.f;
        if ( h > 1.f ) h -= 1.f;
        ImGui::ColorConvertHSVtoRGB( h, 1.f, 1.f, r, g, b );
        data.pixels[i + resX] = Color( r, g, b );
    }
    data.filter = FilterType::Linear;
    textureR->update( data );

}

std::unique_ptr<MR::ImGuiImage>& RibbonButtonDrawer::GetTexture( TextureType type )
{
    const int typeInt = int( type );
    assert( typeInt < textures_.size() && typeInt >= 0 );
    return textures_[typeInt];
}


bool RibbonButtonDrawer::GradientCheckboxItem( const MenuItemInfo& item, bool* value ) const
{
    bool res = UI::checkbox( ( "##" + item.item->name() ).c_str(), value );
    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x + 3;
    ImGui::SameLine( 0.f, spacing );
    const float height = ImGui::GetTextLineHeight();
    drawButtonIcon( item, DrawButtonParams{.itemSize = ImVec2( height + 4, height + 4 ), .iconSize = height / scaling_,
                                           .rootType = DrawButtonParams::RootType::Toolbar } );
    ImGui::SameLine( 0.f, spacing );
    std::string name = item.caption.empty() ? item.item->name() : item.caption;
    ImGui::Text( "%s", name.c_str());
    return res;
}

bool RibbonButtonDrawer::CustomCollapsingHeader( const char* label, ImGuiTreeNodeFlags flags, int issueCount )
{
    const bool bulletMode = bool( flags & ImGuiTreeNodeFlags_Bullet );
    const auto& style = ImGui::GetStyle();
    auto pos = ImGui::GetCursorScreenPos();
    pos.x += style.FramePadding.x;
    pos.y += style.FramePadding.y;

    auto context = ImGui::GetCurrentContext();
    auto window = context->CurrentWindow;
    auto drawList = window->DrawList;

    const float height = ImGui::GetTextLineHeight();
    const float width = ImGui::GetTextLineHeight();
    const float textWidth = ImGui::CalcTextSize( label ).x;

    if ( auto forcedState = UI::TestEngine::createValueTentative<bool>( label ) )
        ImGui::SetNextItemOpen( *forcedState );

    bool res = ImGui::CollapsingHeader( label, flags );

    (void)UI::TestEngine::createValue( label, res, false, true );

    for ( int i = 0; i < issueCount; ++i )
    {
        drawList->AddCircleFilled( { pos.x + textWidth + 3.0f * width + i * width, pos.y + height / 2.0f }, height / 3.0f, Color( 0.886f, 0.267f, 0.267f, 1.0f ).getUInt32() );
    }

    const auto isActive = ImGui::IsItemActive();
    bool setOverlap = false;
    if ( bool( flags & ImGuiTreeNodeFlags_AllowOverlap ) )
    {
        setOverlap = true;
        ImGui::GetCurrentContext()->LastItemData.InFlags |= ImGuiItemFlags_AllowOverlap;
    }
    const auto isHovered = ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenBlockedByActiveItem );
    if ( setOverlap )
    {
        ImGui::GetCurrentContext()->LastItemData.InFlags &= ( ~ImGuiItemFlags_AllowOverlap );
    }

    const auto windowBgColor = ImGui::GetStyleColorVec4( ImGuiCol_WindowBg );
    const auto headerColor = ImGui::GetStyleColorVec4( ( isActive && isHovered ) ? ImGuiCol_HeaderActive : isHovered ? ImGuiCol_HeaderHovered : ImGuiCol_Header );
    const float alpha = headerColor.w;

    const ImVec4 blendedHeaderColor
    {
        windowBgColor.x + ( headerColor.x - windowBgColor.x ) * alpha,
        windowBgColor.y + ( headerColor.y - windowBgColor.y ) * alpha,
        windowBgColor.z + ( headerColor.z - windowBgColor.z ) * alpha,
        1.0f
    };

    drawList->AddRectFilled( pos, { pos.x + width, pos.y + height }, ImGui::GetColorU32( blendedHeaderColor ) );
    const float thickness = ImMax( height * 0.15f, 1.0f );
    const auto halfHeight = height * 0.5f;
    const auto halfWidth = width * 0.5f;
    if ( bulletMode )
    {
        const ImVec2 center{ pos.x + halfWidth,pos.y + halfHeight };
        drawList->AddCircleFilled( center, thickness * 1.0f, ImGui::GetColorU32( ImGuiCol_Text ) );
    }
    else
    {
        if ( res )
        {
            const auto horIndent = height * 0.25f;
            const auto vertIndent = height * 7.5f / 20.0f;
            const ImVec2 startPoint{ pos.x + horIndent, pos.y + vertIndent };
            const ImVec2 midPoint{ pos.x + halfWidth, pos.y + height - vertIndent };
            const ImVec2 endPoint{ pos.x + width - horIndent, pos.y + vertIndent };

            DrawCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
        }
        else
        {
            const auto horIndent = width * 7.5f / 20.0f;
            const auto vertIndent = height * 0.25f;

            const ImVec2 startPoint{ pos.x + horIndent, pos.y + vertIndent };
            const ImVec2 midPoint{ pos.x + width - horIndent, pos.y + halfHeight };
            const ImVec2 endPoint{ pos.x + horIndent, pos.y + height - vertIndent };

            DrawCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
        }
    }
    return res;
}

RibbonButtonDrawer::ButtonItemWidth RibbonButtonDrawer::calcItemWidth( const MenuItemInfo& item, DrawButtonParams::SizeType sizeType ) const
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

void RibbonButtonDrawer::drawButtonItem( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    CustomButtonParameters cParams;
    cParams.iconType = RibbonIcons::IconType::RibbonItemIcon;
    drawCustomButtonItem( item, cParams, params );
}

void RibbonButtonDrawer::drawCustomButtonItem( const MenuItemInfo& item, const CustomButtonParameters& customParam,
    const DrawButtonParams& params ) const
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

    int colorChanged = customParam.pushColorsCb ?
        customParam.pushColorsCb( requirements.empty(), item.item->isActive() ) :
        pushRibbonButtonColors( requirements.empty(), item.item->isActive(), params.forceHovered, params.rootType );
    ImGui::SetNextItemAllowOverlap();
    bool pressed = ImGui::ButtonEx( ( "##wholeChildBtn" + item.item->name() ).c_str(), itemSize, ImGuiButtonFlags_AllowOverlap );
    pressed = UI::TestEngine::createButton( item.item->name() ) || pressed; // Must not short-circuit.
    pressed = pressed || params.forcePressed;

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    float fontScale = 1.f;
    if ( font ) {
        if ( params.iconSize != 0 )
            font->Scale = params.iconSize / cBigIconSize;
        else if ( params.sizeType != DrawButtonParams::SizeType::Big )
            font->Scale = cSmallIconSize / cBigIconSize;
        fontScale = font->Scale;
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * fontScale * scaling_ );
    ImVec2 iconRealSize = ImVec2( imageRequiredSize, imageRequiredSize );
    bool needTextColor = !requirements.empty() || item.item->isActive() || params.rootType != DrawButtonParams::Ribbon;
    bool needChangeColor = needTextColor || monochrome_.has_value();
    auto* imageIcon = RibbonIcons::findByName( item.item->name(), iconRealSize.x, needChangeColor ?
                                               RibbonIcons::ColorType::White : RibbonIcons::ColorType::Colored,
                                               customParam.iconType );

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
            ImGui::SetCursorPosY( 3 * scaling_ + 2.0f * ImGui::GetStyle().WindowPadding.y );
        else
            ImGui::SetCursorPosY( 3 * scaling_ + ImGui::GetStyle().WindowPadding.y );
    }

    if ( !imageIcon )
    {
        ( !needTextColor && monochrome_.has_value() ) ?
            ImGui::TextColored( ImVec4( Vector4f( *monochrome_ ) ), "%s", item.icon.c_str() ) :
            ImGui::Text( "%s", item.icon.c_str() );
    }
    else
    {
        ImVec4 multColor = ImVec4( 1, 1, 1, 1 );
        if ( needChangeColor )
        {
            multColor = ( !needTextColor && monochrome_.has_value() ) ?
                ImVec4( Vector4f( *monochrome_ ) ) :
                ImGui::GetStyleColorVec4( ImGuiCol_Text );
        }
        ImGui::Image( *imageIcon, iconRealSize, multColor );
    }

    if ( font )
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
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f + 3 * scaling_ );
        else
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f - ImGui::GetStyle().WindowPadding.y + 3 * scaling_ );

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
    {
        onPressAction_( item.item, requirements );
        if ( params.isPressed )
            *params.isPressed = true;
    }

    if ( ImGui::IsItemHovered() )
        drawTooltip_( item, requirements );

    if ( dropItem )
        drawButtonDropItem_( item, params );
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

void RibbonButtonDrawer::drawButtonIcon( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    ImGui::BeginGroup();

    int colorChanged = pushRibbonButtonColors( true, false, params.forceHovered, params.rootType );

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    float fontScale = 1.f;
    if ( font )
    {
        if ( params.iconSize != 0 )
            font->Scale = params.iconSize / cBigIconSize;
        else if ( params.sizeType != DrawButtonParams::SizeType::Big )
            font->Scale = cSmallIconSize / cBigIconSize;
        fontScale = font->Scale;
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * fontScale * scaling_ );
    ImVec2 iconRealSize = ImVec2( imageRequiredSize, imageRequiredSize );
    bool needTextColor = params.rootType != DrawButtonParams::Ribbon;
    bool needChangeColor = needTextColor || monochrome_.has_value();
    auto* imageIcon = RibbonIcons::findByName( item.item->name(), iconRealSize.x, needChangeColor ?
                                               RibbonIcons::ColorType::White : RibbonIcons::ColorType::Colored,
                                               RibbonIcons::IconType::RibbonItemIcon );

    if ( !imageIcon )
        iconRealSize = ImGui::CalcTextSize( item.icon.c_str() );

    ImVec2 cursorPos = ImGui::GetCursorPos();
    cursorPos.x += ( params.itemSize.x - iconRealSize.x ) / 2.f;
    cursorPos.y += ( params.itemSize.y - iconRealSize.y ) / 2.f;
    ImGui::SetCursorPos( cursorPos );

    if ( !imageIcon )
    {
        ( !needTextColor && monochrome_.has_value() ) ? 
            ImGui::TextColored( ImVec4( Vector4f( *monochrome_ ) ), "%s", item.icon.c_str() ) : 
            ImGui::Text( "%s", item.icon.c_str() );
    }
    else
    {
        ImVec4 multColor = ImVec4( 1, 1, 1, 1 );
        if ( needChangeColor )
        {
            multColor = ( !needTextColor && monochrome_.has_value() ) ?
                ImVec4( Vector4f( *monochrome_ ) ) :
                ImGui::GetStyleColorVec4( ImGuiCol_Text );
        }
        ImGui::Image( *imageIcon, iconRealSize, multColor );
    }

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    if ( colorChanged > 0 )
        ImGui::PopStyleColor( colorChanged );

    ImGui::EndGroup();
}

bool RibbonButtonDrawer::drawTabArrowButton( const char* icon, const ImVec2& size, float iconSize )
{
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrab ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( font )
    {
        font->Scale = iconSize / RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );
    }

    bool pressed = ImGui::Button( icon, size );

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar();
    return pressed;
}

void RibbonButtonDrawer::setMonochrome( const std::optional<Color>& color )
{
    monochrome_ = color;
}

void RibbonButtonDrawer::drawButtonDropItem_( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    float iconSize = params.iconSize * 0.5f;
    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( font )
    {
        const float fontSize = RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( params.sizeType == DrawButtonParams::SizeType::Big )
            font->Scale = iconSize / fontSize;
        else
            font->Scale = iconSize * 1.5f / fontSize;
        ImGui::PushFont( font );
    }
    auto frameHeight = ImGui::GetFrameHeight();
    ImVec2 itemSize = ImVec2( frameHeight, frameHeight );
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
        itemSize.y = params.itemSize.y;
        dropBtnPos.x = params.itemSize.x - itemSize.x;
        dropBtnPos.y = 0.0f;
    }
    ImGui::SetCursorPos( dropBtnPos );
    auto absMinPos = ImGui::GetCurrentContext()->CurrentWindow->DC.CursorPos;

    auto name = "##DropDown" + item.item->name();
    auto nameWindow = name + "Popup";
    bool menuOpened = ImGui::IsPopupOpen( nameWindow.c_str() );

    bool dropBtnEnabled = !item.item->dropItems().empty();

    int pushedColors = pushRibbonButtonColors( dropBtnEnabled, menuOpened, params.forceHovered, params.rootType );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    bool comboPressed = ( ImGui::Button( name.c_str(), itemSize ) || UI::TestEngine::createButton( name ) ) && dropBtnEnabled;

    auto iconRealSize = ImGui::CalcTextSize( "\xef\x81\xb8" ); //down icon
    ImGui::SetCursorPosX( dropBtnPos.x + ( itemSize.x - iconRealSize.x + 1 ) * 0.5f );
    if ( params.sizeType == DrawButtonParams::SizeType::Big )
        ImGui::SetCursorPosY( params.itemSize.y - frameHeight + ( frameHeight - iconRealSize.y - 1 ) * 0.5f );
    else
        ImGui::SetCursorPosY( dropBtnPos.y + ( itemSize.y - iconRealSize.y - 1 ) * 0.5f );

    ImGui::Text( "%s", "\xef\x81\xb8" );

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( pushedColors );

    if ( font )
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
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + itemSize.y );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowSizeConstraints( ImVec2(), ImVec2( -1, 200 * scaling_ ) );
    ImGui::Begin( nameWindow.c_str(), NULL, window_flags );
    if ( menuOpened )
    {
        UI::TestEngine::pushTree( item.item->name() + "##DropDownList" );
        MR_FINALLY{ UI::TestEngine::popTree(); };

        drawDropList_( item.item );
        ImGui::EndPopup();
    }
}

void RibbonButtonDrawer::drawDropList_( const std::shared_ptr<RibbonMenuItem>& baseDropItem ) const
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
        if ( it == schema.items.end() )
        {
            auto pressed = ImGui::MenuItem( ( caption + "##dropItem" ).c_str(), nullptr, dropItem->isActive(), requirements.empty() );
            if ( pressed )
                onPressAction_( dropItem, requirements );
        }
        else
        {
            const auto& item = it->second;

            if ( !item.caption.empty() )
                caption = item.caption;

            const auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * scaling_;
            const auto width = calcItemWidth( item, DrawButtonParams::SizeType::SmallText );

            DrawButtonParams params;
            params.sizeType = DrawButtonParams::SizeType::SmallText;
            params.iconSize = cSmallIconSize;
            params.itemSize.y = ySize;
            params.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * scaling_;
            drawButtonItem( item, params );
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

void RibbonButtonDrawer::drawTooltip_( const MenuItemInfo& item, const std::string& requirements ) const
{
    auto sFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small );
    if ( sFont )
        ImGui::PushFont( sFont );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 0 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cRibbonButtonWindowPaddingX * scaling_, cRibbonButtonWindowPaddingY * scaling_ ) );
    std::string tooltip = item.item->getDynamicTooltip();
    if ( tooltip.empty() )
        tooltip = item.tooltip;

    const auto& caption = item.caption.empty() ? item.item->name() : item.caption;

    std::string fullText;
    fullText = caption;
    std::string shortcutStr;

    if ( shortcutManager_ )
    {
        auto shortcut = shortcutManager_->findShortcutByName( item.item->name() );
        if ( shortcut )
        {
            shortcutStr = " (" + ShortcutManager::getKeyFullString( *shortcut ) + ")";
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
    if ( sFont )
        ImGui::PopFont();
}


int RibbonButtonDrawer::pushRibbonButtonColors( bool enabled, bool active, bool forceHovered, DrawButtonParams::RootType rootType ) const
{
    if ( active )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveClicked ).getUInt32() );
        if ( !forceHovered )
            ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActive ).getUInt32() );
        else
            ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveHovered ).getUInt32() );
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

    if ( !forceHovered )
        ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
    if ( rootType == DrawButtonParams::RootType::Ribbon )
    {
        if ( forceHovered )
            ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonClicked ).getUInt32() );
    }
    else if ( rootType == DrawButtonParams::RootType::Toolbar )
    {
        if ( forceHovered )
            ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
    }
    else
    {
        assert( rootType == DrawButtonParams::RootType::Header );
        if ( forceHovered )
            ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabClicked ).getUInt32() );
    }
    return 4;
}

}
