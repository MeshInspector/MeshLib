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
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::Button( label, size );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientButtonCommonSize( const char* label )
{
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );
    const ImGuiStyle& style = ImGui::GetStyle();
    auto framePadding = style.FramePadding;
    if ( auto menu = getViewerInstance().getMenuPlugin() )
        framePadding.y *= menu->menu_scaling();
    return GradientButton( label, ImVec2( 0, labelSize.y + framePadding.y * 2.0f ) );
}

bool RibbonButtonDrawer::GradientButtonValid( const char* label, bool valid, const ImVec2& size /* = ImVec2(0, 0) */ )
{
    auto& texture = GetGradientTexture();
    if ( !texture )
        return ImGui::ButtonValid( label, valid, size );

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
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::ButtonValid( label, valid, size );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientCheckbox( const char* label, bool* value )
{
    auto& texture = GetGradientTexture();
    if ( !texture || ( value && !*value ) )
        return  ImGui::Checkbox( label, value );

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
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    //code of this lambda is copied from ImGui::Checkbox in order to decrease thickness and change appearance of the check mark
    auto drawCustomCheckbox = [] ( const char* label, bool* v )
    {
        if ( !ImGui::GetCurrentContext() )
            return false;

        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = g.CurrentWindow;
        if ( !window || window->SkipItems )
            return false;

        const ImGuiStyle& style = ImGui::GetStyle();
        const ImGuiID id = window->GetID( label );
        const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );

        const float square_sz = ImGui::GetFrameHeight();
        const ImVec2 pos = window->DC.CursorPos;
        const ImRect total_bb( pos, ImVec2( pos.x + square_sz + ( label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f ), pos.y + label_size.y + style.FramePadding.y * 2.0f ) );
        ImGui::ItemSize( total_bb, style.FramePadding.y );
        if ( !ImGui::ItemAdd( total_bb, id ) )
        {
            IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Checkable | ( *v ? ImGuiItemStatusFlags_Checked : 0 ) );
            return false;
        }

        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( total_bb, id, &hovered, &held );
        if ( pressed )
        {
            *v = !( *v );
            ImGui::MarkItemEdited( id );
        }

        const ImRect check_bb( pos, ImVec2( pos.x + square_sz, pos.y + square_sz ) );
        ImGui::RenderNavHighlight( total_bb, id );
        ImGui::RenderFrame( check_bb.Min, check_bb.Max, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg ), true, style.FrameRounding );
        ImU32 check_col = ImGui::GetColorU32( ImGuiCol_CheckMark );
        bool mixed_value = ( g.LastItemData.InFlags & ImGuiItemFlags_MixedValue ) != 0;
        if ( mixed_value )
        {
            // Undocumented tristate/mixed/indeterminate checkbox (#2644)
            // This may seem awkwardly designed because the aim is to make ImGuiItemFlags_MixedValue supported by all widgets (not just checkbox)
            ImVec2 pad( ImMax( 1.0f, IM_FLOOR( square_sz / 3.6f ) ), ImMax( 1.0f, IM_FLOOR( square_sz / 3.6f ) ) );
            window->DrawList->AddRectFilled( { check_bb.Min.x + pad.x,  check_bb.Min.y + pad.y }, { check_bb.Max.x - pad.x, check_bb.Max.y - pad.y }, check_col, style.FrameRounding );
        }
        else if ( *v )
        {
            const float pad = ImMax( 1.0f, IM_FLOOR( square_sz / 6.0f ) );
            auto renderCustomCheckmark = [] ( ImDrawList* draw_list, ImVec2 pos, ImU32 col, float sz )
            {
                const float thickness = ImMax( sz * 0.15f, 1.0f );
                sz -= thickness * 0.5f;
                pos = ImVec2( pos.x + thickness * 0.25f, pos.y + thickness * 0.25f );

                const float half = sz * 0.5f;
                const float ninth = sz / 9.0f;
                const ImVec2 startPoint { pos.x + ninth, pos.y + half };
                const ImVec2 anglePoint { pos.x + half, pos.y + sz - ninth };
                const ImVec2 endPoint { pos.x + sz - ninth, pos.y + ninth * 2.0f };

                draw_list->PathLineTo( startPoint );
                draw_list->PathLineTo( anglePoint );
                draw_list->PathLineTo( endPoint );
                draw_list->PathStroke( col, 0, thickness );

                const float radius = thickness * 0.5f;
                draw_list->AddCircleFilled( startPoint, radius, col );
                draw_list->AddCircleFilled( anglePoint, radius, col );
                draw_list->AddCircleFilled( endPoint, radius, col );
            };
            renderCustomCheckmark( window->DrawList, { check_bb.Min.x +  pad, check_bb.Min.y + pad }, check_col, square_sz - pad * 2.0f );
        }

        ImVec2 label_pos = ImVec2( check_bb.Max.x + style.ItemInnerSpacing.x, check_bb.Min.y + style.FramePadding.y );
        if ( g.LogEnabled )
            ImGui::LogRenderedText( &label_pos, mixed_value ? "[~]" : *v ? "[x]" : "[ ]" );
        if ( label_size.x > 0.0f )
            ImGui::RenderText( label_pos, label );

        IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Checkable | ( *v ? ImGuiItemStatusFlags_Checked : 0 ) );
        return pressed;
    };

    auto res = drawCustomCheckbox( label, value );

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
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), clickSize * 0.5f );

    //code of this lambda is copied from ImGui::RadioBitton in order to decrease size of the central circle
    auto drawCustomRadioButton = []( const char* label, int* v, int v_button )
    {     
        if ( !ImGui::GetCurrentContext() )
            return false;

        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = g.CurrentWindow;
        if ( !window || window->SkipItems )
            return false;

        const ImGuiStyle& style = ImGui::GetStyle();
        const ImGuiID id = window->GetID( label );
        const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );

        const float square_sz = ImGui::GetFrameHeight();
        const ImVec2 pos = window->DC.CursorPos;
        const ImRect check_bb( pos, ImVec2( pos.x + square_sz, pos.y + square_sz ) );
        const ImRect total_bb( pos, ImVec2( pos.x + square_sz + ( label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f ), pos.y + label_size.y + style.FramePadding.y * 2.0f ) );
        ImGui::ItemSize( total_bb, style.FramePadding.y );
        if ( !ImGui::ItemAdd( total_bb, id ) )
            return false;

        ImVec2 center = check_bb.GetCenter();
        const float radius = ( square_sz - 1.0f ) * 0.5f;

        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( total_bb, id, &hovered, &held );
        if ( pressed )
            ImGui::MarkItemEdited( id );

        ImGui::RenderNavHighlight( total_bb, id );
        window->DrawList->AddCircleFilled( center, radius, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg ), 16 );

        const bool active = *v == v_button;
        if ( active )
        {
            const float pad = ImMax( 1.0f, IM_FLOOR( square_sz * 0.3f ) );
            window->DrawList->AddCircleFilled( center, radius - pad, ImGui::GetColorU32( ImGuiCol_CheckMark ), 16 );
        }

        if ( style.FrameBorderSize > 0.0f )
        {
            window->DrawList->AddCircle( ImVec2( center.x + 1, center.y + 1 ), radius, ImGui::GetColorU32( ImGuiCol_BorderShadow ), 16, style.FrameBorderSize );
            window->DrawList->AddCircle( center, radius, ImGui::GetColorU32( ImGuiCol_Border ), 16, style.FrameBorderSize );
        }

        ImVec2 label_pos = ImVec2( check_bb.Max.x + style.ItemInnerSpacing.x, check_bb.Min.y + style.FramePadding.y );
        ImGui::RenderText( label_pos, label );

        IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags );
        return pressed;
    };

    auto res = drawCustomRadioButton( label, value, v_button );

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::CustomCombo( const char* label, int* v, const std::vector<std::string>& options, bool showPreview, const std::vector<std::string>& tooltips )
{
    assert( tooltips.empty() || tooltips.size() == options.size() );

    auto context = ImGui::GetCurrentContext();
    ImGuiWindow* window = context->CurrentWindow;
    const auto& style = ImGui::GetStyle();
    const ImVec2 pos = window->DC.CursorPos;
    const float arrowSize = 2 * style.FramePadding.y + ImGui::GetTextLineHeight();
    if ( !showPreview )
        ImGui::PushItemWidth( arrowSize + style.FramePadding.x * 0.5f );

    float itemWidth = ( context->NextItemData.Flags & ImGuiNextItemDataFlags_HasWidth ) ? context->NextItemData.Width : window->DC.ItemWidth;

    auto res = ImGui::BeginCombo( label , showPreview ? options[*v].c_str() : nullptr, ImGuiComboFlags_NoArrowButton );
    
    const ImRect boundingBox( pos, { pos.x + itemWidth, pos.y +  arrowSize } );
    const ImRect arrowBox( { pos.x + boundingBox.GetWidth() - boundingBox.GetHeight() * 6.0f / 7.0f, pos.y }, boundingBox.Max );

    auto renderCustomArrow = [] ( ImDrawList* draw_list, ImRect arrowBox, ImU32 col )
    {
        const float halfHeight = arrowBox.GetHeight() * 0.5f;
        const float seventhHeight = arrowBox.GetHeight() / 7.0f;
        const float sixthWidth = arrowBox.GetWidth() / 6.0f;
            
        const float thickness = ImMax( arrowBox.GetHeight() * 0.1f, 1.0f );
        const ImVec2 pos { arrowBox.Min.x, arrowBox.Min.y - thickness };
        const ImVec2 startPoint { pos.x + sixthWidth, pos.y + halfHeight };
        const ImVec2 anglePoint { pos.x + 2 * sixthWidth, pos.y + halfHeight + seventhHeight };
        const ImVec2 endPoint { pos.x + 3 * sixthWidth, pos.y + halfHeight };

        draw_list->PathLineTo( startPoint );       
        draw_list->PathLineTo( anglePoint );
        draw_list->PathLineTo( endPoint );        
        draw_list->PathStroke( col, 0, thickness );

        const float radius = thickness * 0.5f;
        draw_list->AddCircleFilled( startPoint, radius, col );
        draw_list->AddCircleFilled( anglePoint, radius, col );
        draw_list->AddCircleFilled( endPoint, radius, col );
    };

    renderCustomArrow( window->DrawList, arrowBox, ImGui::GetColorU32( ImGuiCol_Text ) );    

    if ( !res )
        return false;
    
    for ( int i = 0; i < int(options.size()); ++i )
    {
        ImGui::PushID( (label + std::to_string( i )).c_str() );
        if ( ImGui::Selectable( options[i].c_str(), *v == i ) )
            *v = i;
        
        if ( !tooltips.empty() )
            ImGui::SetTooltipIfHovered( tooltips[i], Viewer::instanceRef().getMenuPlugin()->menu_scaling() );

        ImGui::PopID();        
    }

    ImGui::EndCombo();
    if ( !showPreview )
        ImGui::PopItemWidth();
    return true;
}

bool RibbonButtonDrawer::CustomCollapsingHeader( const char* label, ImGuiTreeNodeFlags flags )
{
    const auto& style = ImGui::GetStyle();
    auto pos = ImGui::GetCursorScreenPos();
    pos.x += style.FramePadding.x;
    pos.y += style.FramePadding.y;

    auto res = ImGui::CollapsingHeader( label, flags );    
    
    const float height = ImGui::GetTextLineHeight();
    const float width = ImGui::GetTextLineHeight();

    const auto isActive = ImGui::IsItemActive();
    const auto isHovered = ImGui::IsItemHovered();

    const auto windowBgColor = ImGui::GetStyleColorVec4( ImGuiCol_WindowBg );
    const auto headerColor = ImGui::GetStyleColorVec4( ( isActive ) ? ImGuiCol_HeaderActive : isHovered ? ImGuiCol_HeaderHovered : ImGuiCol_Header );
    const float alpha = headerColor.w;

    const ImVec4 blendedHeaderColor
    {
        windowBgColor.x + ( headerColor.x - windowBgColor.x ) * alpha,
        windowBgColor.y + ( headerColor.y - windowBgColor.y ) * alpha,
        windowBgColor.z + ( headerColor.z - windowBgColor.z ) * alpha,
        1.0f
    };
    
    auto context = ImGui::GetCurrentContext();
    auto window = context->CurrentWindow;
    auto drawList = window->DrawList;

    drawList->AddRectFilled( pos, { pos.x + width, pos.y + height }, ImGui::GetColorU32( blendedHeaderColor ) );

    auto renderCustomArrow = [] ( ImDrawList* drawList, const ImVec2& startPoint, const ImVec2& midPoint, const ImVec2& endPoint, ImU32 col, float thickness )
    {
        drawList->PathLineTo( startPoint );
        drawList->PathLineTo( midPoint );
        drawList->PathLineTo( endPoint );
        drawList->PathStroke( col, 0, thickness );

        const float radius = thickness * 0.5f;
        drawList->AddCircleFilled( startPoint, radius, col );
        drawList->AddCircleFilled( midPoint, radius, col );
        drawList->AddCircleFilled( endPoint, radius, col );
    };

    const float thickness = ImMax( height * 0.2f, 1.0f );
    if ( res )
    {
        const auto halfWidth = width * 0.5f;
        const auto horIndent = height * 0.2f;
        const auto vertIndent = height * 7.0f / 20.0f;

        const ImVec2 startPoint { pos.x + horIndent, pos.y + vertIndent };
        const ImVec2 midPoint{ pos.x + halfWidth, pos.y + height - vertIndent };
        const ImVec2 endPoint{ pos.x + width - horIndent, pos.y + vertIndent };

        renderCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
    }
    else
    {
        const auto halfHeight = height * 0.5f;
        const auto horIndent = width * 7.0f / 20.0f;
        const auto vertIndent = height * 0.2f;

        const ImVec2 startPoint{ pos.x + horIndent, pos.y + vertIndent };
        const ImVec2 midPoint{ pos.x + width - horIndent, pos.y + halfHeight };
        const ImVec2 endPoint{ pos.x + horIndent, pos.y + height - vertIndent };

        renderCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
    }

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

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    float fontScale = 1.f;
    if ( font ) {
        const float iconSize = RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( iconSize )
        {
            if ( params.iconSize != 0 )
                font->Scale = params.iconSize / iconSize;
            else if ( params.sizeType != DrawButtonParams::SizeType::Big )
                font->Scale = cSmallIconSize / iconSize;
        }
        fontScale = font->Scale;
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * fontScale * scaling_ );
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
    auto iconRealSize = ImGui::CalcTextSize( "\xef\x81\xb8" ); //down icon
    ImGui::SetCursorPosX( dropBtnPos.x + ( itemSize.x - iconRealSize.x + 1 ) * 0.5f );
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
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
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
