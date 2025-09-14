#include "MRDrawViewportWidgetsItem.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRSceneColors.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRColorTheme.h"
#include "MRViewer/MRImGuiImage.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

#include <imgui.h>

#include <span>


namespace MR
{

using namespace ImGuiMath;

MR_REGISTER_RIBBON_ITEM( DrawViewportWidgetsItem )

ProvidesViewportWidget::ProvidesViewportWidget()
{
    providedWidgetsConnection_ = DrawViewportWidgetsItem::getHandleViewportSignal_().connect(
        [this]( ProvidesViewportWidget::ViewportWidgetInterface& in )
        {
            providedViewportWidgets( in );
        }
    );
}

DrawViewportWidgetsItem::DrawViewportWidgetsItem()
    : RibbonMenuItem( "Draw Viewport Widgets" )
{
    preDrawConnection_ = getViewerInstance().preDrawSignal.connect( [this]
    {
        for ( Viewport& viewport : getViewerInstance().viewport_list )
            handleViewport( viewport );
    } );
}

void DrawViewportWidgetsItem::handleViewport( Viewport& viewport )
{
    struct Entry
    {
        float order = 0;
        std::string name;
        bool active = false;
        std::string icon;
        std::function<void()> onClick;

        auto tieForSorting() const { return std::tie( order, name ); }
    };

    struct Impl : ProvidesViewportWidget::ViewportWidgetInterface
    {
        ViewportId id;
        std::vector<Entry> entries;

        [[nodiscard]] ViewportId viewportId() const override { return id; }

        void addButton( float order, std::string name, bool active, std::string icon, std::function<void()> onClick ) override
        {
            Entry& e = entries.emplace_back();;
            e.order = order;
            e.name = std::move( name );
            e.active = active;
            e.icon = std::move( icon );
            e.onClick = std::move( onClick );
        }
    };
    Impl impl;
    impl.id = viewport.id;

    getHandleViewportSignal_()( impl );

    std::stable_sort( impl.entries.begin(), impl.entries.end(), []( const Entry& a, const Entry& b )
    {
        return a.tieForSorting() > b.tieForSorting(); // Sort backwards, since we'll draw this right-to-left.
    } );

    Box2f rect = viewport.getViewportRect();
    // Flip the vertical axis to point down.
    rect.min.y = ImGui::GetIO().DisplaySize.y - rect.min.y;
    rect.max.y = ImGui::GetIO().DisplaySize.y - rect.max.y;
    std::swap( rect.min.y, rect.max.y );

    const float menuScaling = getViewerInstance().getMenuPlugin()->menu_scaling();
    const ImVec2 spacingToCorner = round( ImVec2( 22, 22 ) * menuScaling );
    const float spacingBetweenButtons = std::round( 16 * menuScaling );

    const ImVec2 buttonSize = round( ImVec2( 34, 34 ) * menuScaling ); // `buttonShrink` gets subtracted from this on both sides.
    const ImVec2 imageSize = round( ImVec2( 24, 24 ) * menuScaling );

    const float buttonShrink = std::round( 1 * menuScaling ); // Need this to avoid button borders being clipped.

    ImVec2 curPos = rect.max - spacingToCorner - buttonSize;

    for ( const Entry& e : impl.entries )
    {
        if ( CompareAny( curPos ) < ImVec2( rect.min + spacingToCorner ) )
            continue; // Not enough space in the viewport to draw this!

        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2{} );
        MR_FINALLY{ ImGui::PopStyleVar(); };
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0 ); // The button does the rounding for itself.
        MR_FINALLY{ ImGui::PopStyleVar(); };

        ImGui::SetNextWindowPos( curPos );
        ImGui::SetNextWindowSize( buttonSize );

        ImGui::Begin( fmt::format( "##cornerButtonWindow.{}.{}", viewport.id.value(), e.name ).c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground );
        MR_FINALLY{ ImGui::End(); };

        // Force the window to stay in background.
        ImGui::BringWindowToDisplayBack( ImGui::GetCurrentWindow() );

        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
        MR_FINALLY{ ImGui::PopStyleVar(); };
        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 7 * menuScaling );
        MR_FINALLY{ ImGui::PopStyleVar(); };
        ImGui::PushStyleColor( ImGuiCol_Button, e.active ? ImGui::GetStyleColorVec4( ImGuiCol_SliderGrab ) : ImVec4( 0, 0, 0, 0 ) );
        MR_FINALLY{ ImGui::PopStyleColor(); };

        ImGui::SetCursorScreenPos( curPos + buttonShrink );

        if ( e.active )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, Color::white() );
            // Everything else remains defaulted.
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Text ) );
            ImGui::PushStyleColor( ImGuiCol_Button, viewport.getParameters().backgroundColor.scaledAlpha( 0.75f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::instance().getRibbonColor( ColorTheme::RibbonColorsType::BackgroundSecStyle ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::instance().getRibbonColor( ColorTheme::RibbonColorsType::HeaderSeparator ) );
        }
        MR_FINALLY{ ImGui::PopStyleColor( e.active ? 1 : 4 ); };

        // The button.
        if ( ImGui::Button( fmt::format( "##cornerButtonWindow.{}.{}", viewport.id.value(), e.name ).c_str(), buttonSize - buttonShrink * 2 ) )
            e.onClick();

        // The icon.
        auto icon = RibbonIcons::findByName( e.icon, std::max( imageSize.x, imageSize.y ), RibbonIcons::ColorType::White, RibbonIcons::IconType::IndependentIcons );
        assert( icon );
        if ( icon )
        {
            ImDrawList& list = *ImGui::GetWindowDrawList();
            ImVec2 imagePos = round( curPos + ( buttonSize - imageSize ) / 2 );
            list.AddImage( icon->getImTextureId(), imagePos, imagePos + imageSize, ImVec2( 0, 1 ), ImVec2( 1, 0 ), ImGui::ColorConvertFloat4ToU32( ImGui::GetStyleColorVec4( ImGuiCol_Text ) ) );
        }

        // Update the position.
        curPos.x -= buttonSize.x + spacingBetweenButtons;
    }
}

DrawViewportWidgetsItem::HandleViewportSignal& DrawViewportWidgetsItem::getHandleViewportSignal_()
{
    static HandleViewportSignal ret; // Needs to be here to avoid the construction order fiasco.
    return ret;
}

}
