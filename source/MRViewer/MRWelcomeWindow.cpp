#include "MRWelcomeWindow.h"
#include "imgui.h"
#include "MRSceneCache.h"
#include "MRUIStyle.h"
#include "MRMesh/MRVisualObject.h"
#include "MRRibbonMenu.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRSystem.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonMenuItem.h"
#include "MRRibbonSchema.h"
#include "MRRibbonIcons.h"
#include "MRColorTheme.h"

namespace MR
{

const float cSubwindowWidth = 425.f;
const float cSubwindowHeight = 230.f;
const float cSubwindowSpacing = 16.f;
const float cWindowWidth = cSubwindowWidth * 2.f + cSubwindowSpacing;
const float cWindowHeight = cSubwindowHeight * 2.f + cSubwindowSpacing;

const float cPaddingLeft = 36.f;


void WelcomeWindow::draw()
{
    if ( !visible_ || !menu_ )
        return;

    if ( !SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selectable>().empty() || menu_->hasAnyActiveItem() )
    {
        visible_ = false;
        return;
    }
    
    scaling_ = menu_->menu_scaling();

    const float sceneWidth = float ( getViewerInstance().framebufferSize.x ) - menu_->getSceneSize().x;
    const float sceneHeight = float( getViewerInstance().framebufferSize.y ) - menu_->getTopPanelCurrentHeight() * scaling_ ;

    if ( ( sceneWidth < ( cSubwindowWidth * 2 + cSubwindowSpacing * 3 ) * scaling_ ) ||
        ( sceneHeight < ( cSubwindowHeight * 2 + cSubwindowSpacing * 5 ) * scaling_ ) )
        return;


    sceneCenter_ = { ( getViewerInstance().framebufferSize.x + menu_->getSceneSize().x ) / 2.f,
        ( getViewerInstance().framebufferSize.y + menu_->getTopPanelCurrentHeight() ) / 2.f };

    ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, cSubwindowSpacing / 2.f );
    drawDragDropArea_();
    drawQuickstart_();
    drawCreateSimpleObject_();
    ImGui::PopStyleVar();

    drawCheckbox_();
}

void WelcomeWindow::init( std::function<void( const std::shared_ptr<RibbonMenuItem>&, bool )> itemPressedFn )
{
    menu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    itemPressedFn_ = itemPressedFn;
}

void WelcomeWindow::setShowOnStartup( bool show, bool applyNow /*= false */ )
{
    if ( showOnStartup_ == show )
        return;
    showOnStartup_ = show;
    if ( applyNow )
        visible_ = show;
}

void WelcomeWindow::drawDragDropArea_()
{
    Vector2f windowsPos( sceneCenter_.x - ( cSubwindowWidth + cSubwindowSpacing / 2.f ) * scaling_,
        sceneCenter_.y - ( cSubwindowHeight + cSubwindowSpacing / 2.f ) * scaling_ );
    Vector2f windowsSize( cSubwindowWidth * scaling_, ( cSubwindowHeight * 2 + cSubwindowSpacing ) * scaling_ );
    if ( !beginSubwindow_( "##DragDropArea", windowsPos, windowsSize ) )
        return;

    const float iconWidth = 128.f;
    const float iconHeight = 128.f;
    ImGui::SetCursorPos( ImVec2( ( cSubwindowWidth - iconWidth ) / 2.f * scaling_, ( cSubwindowHeight - iconHeight ) * scaling_ ) );
    auto icon = RibbonIcons::findByName( "DragDrop", iconWidth, RibbonIcons::ColorType::White, RibbonIcons::IconType::IndependentIcons );
    assert( icon );
    ImGui::Image( *icon, { iconWidth * scaling_, iconHeight * scaling_ }, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveText ) );

    const auto& fontManager = menu_->getFontManager();
    auto font = fontManager.getFontByType( RibbonFontManager::FontType::Headline );
    if ( font )
        ImGui::PushFont( font );
    const char title1[] = "Drag & Drop Single File";
    const float titleWidth1 = ImGui::CalcTextSize( title1 ).x;
    ImGui::SetCursorPosY( ( cSubwindowHeight * 1.1f + cSubwindowSpacing ) * scaling_ );
    ImGui::SetCursorPosX( ( cSubwindowWidth * scaling_ - titleWidth1 ) / 2.f );
    ImGui::Text( title1 );
    const char title2[] = "Here to Open";
    const float titleWidth2 = ImGui::CalcTextSize( title2 ).x;
    ImGui::SetCursorPosX( ( cSubwindowWidth * scaling_ - titleWidth2 ) / 2.f );
    ImGui::Text( title2 );
    if ( font )
        ImGui::PopFont();

    ImGui::SetCursorPosY( ( cSubwindowHeight * 1.5f + cSubwindowSpacing ) * scaling_ );
    const char text1[] = "To open DICOMs, RAW Voxels, TIFF use";
    const float textWidth1 = ImGui::CalcTextSize( text1 ).x;
    ImGui::SetCursorPosX( ( cSubwindowWidth * scaling_ - textWidth1 ) / 2.f );
    ImGui::Text( text1 );
    const char text2[] = "ribbon menu.";
    const float textWidth2 = ImGui::CalcTextSize( text2 ).x;
    ImGui::SetCursorPosX( ( cSubwindowWidth * scaling_ - textWidth2 ) / 2.f );
    ImGui::Text( text2 );

    ImGui::End();
}

void WelcomeWindow::drawQuickstart_()
{
    Vector2f windowsPos( sceneCenter_.x + cSubwindowSpacing / 2.f * scaling_,
        sceneCenter_.y - ( cSubwindowHeight + cSubwindowSpacing / 2.f ) * scaling_ );
    Vector2f windowsSize( cSubwindowWidth * scaling_, cSubwindowHeight * scaling_ );
    if ( !beginSubwindow_( "##Quickstart", windowsPos, windowsSize ) )
        return;

    ImGui::SetCursorPos( ImVec2( cPaddingLeft * scaling_, cSubwindowHeight * scaling_ * 0.1f ) );
    const auto& fontManager = menu_->getFontManager();
    auto font = fontManager.getFontByType( RibbonFontManager::FontType::Headline );
    if ( font )
        ImGui::PushFont( font );
    ImGui::Text( "Quickstart" );
    if ( font )
        ImGui::PopFont();

    ImGui::SetCursorPosY( cSubwindowHeight * scaling_ * 0.3f );
    ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "This curated video collection will" );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "quickly and efficiently guide you" );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "through the essentials." );
    ImGui::PopStyleColor();

    const float iconWidth = 130.f;
    const float iconHeight = 128.f;
    ImGui::SetCursorPos( ImVec2( ( cSubwindowWidth - iconWidth * 0.9f ) * scaling_, ( cSubwindowHeight * 0.74f + 36.f - iconHeight + 17.f ) * scaling_ ) );
    auto icon = RibbonIcons::findByName( "Youtube", iconWidth, RibbonIcons::ColorType::White, RibbonIcons::IconType::IndependentIcons );
    assert( icon );
    ImGui::Image( *icon, { iconWidth * scaling_, iconHeight * scaling_ }, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveText ) );
    
    ImGui::SetCursorPos( ImVec2( cPaddingLeft * scaling_, cSubwindowHeight * scaling_ * 0.74f ) );
    if ( UI::button( "Watch Video", { 110 * scaling_, 36 * scaling_ } ) )
        OpenLink( "https://meshinspector.com/inappvideo/onboarding_first_session" );

    ImGui::End();
}

void WelcomeWindow::drawCreateSimpleObject_()
{
    Vector2f windowsPos( sceneCenter_.x + cSubwindowSpacing / 2.f * scaling_,
        sceneCenter_.y + cSubwindowSpacing / 2.f * scaling_ );
    Vector2f windowsSize( cSubwindowWidth * scaling_, cSubwindowHeight * scaling_ );
    if ( !beginSubwindow_( "##CreateSimpleObject", windowsPos, windowsSize ) )
        return;

    ImGui::SetCursorPos( ImVec2( cPaddingLeft * scaling_, cSubwindowHeight * scaling_ * 0.1f ) );
    const auto& fontManager = menu_->getFontManager();
    auto font = fontManager.getFontByType( RibbonFontManager::FontType::Headline );
    if ( font )
        ImGui::PushFont( font );
    ImGui::Text( "Create Simple Object" );
    if ( font )
        ImGui::PopFont();

    ImGui::SetCursorPosY( cSubwindowHeight * scaling_ * 0.3f );
    ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "Easily generate basic shapes like" );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "cubes, spheres, and more using" );
    ImGui::SetCursorPosX( cPaddingLeft * scaling_ );
    ImGui::Text( "default settings." );
    ImGui::PopStyleColor();

    const float sizeIcon = 16 * scaling_;
    const float btnWidth = 64.f;
    const float btnHeight = 50.f;
    const float btnSpacing = 8.f;
    auto drawButton = [&] ( int row, int col, std::string iconName, std::string text )
    {
        const ImVec2 btnPos( ( cSubwindowWidth - btnWidth * ( 2 - col ) - btnSpacing * ( 3 - col ) ) * scaling_,
            ( cSubwindowHeight * 0.74f + 36.f - btnHeight * ( 3 - row ) - btnSpacing * ( 2 - row ) ) * scaling_ );
        ImGui::SetCursorPos( btnPos );
        UI::buttonIconFlatBG( iconName, { sizeIcon, sizeIcon }, text, { btnWidth * scaling_, btnHeight * scaling_ } );
    };
    ImGui::PushStyleColor( ImGuiCol_Button, Color::transparent().getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color::transparent().getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color::transparent().getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
    drawButton( 0, 0, "cube", "Cube" );
    drawButton( 0, 1, "cylinder", "Cylinder" );
    drawButton( 1, 0, "sphere", "Sphere" );
    drawButton( 1, 1, "prism", "Prism" );
    drawButton( 2, 0, "torus", "Torus" );
    drawButton( 2, 1, "arrow", "Arrow" );
    ImGui::PopStyleColor( 4 );

    ImGui::SetCursorPos( ImVec2( cPaddingLeft * scaling_, cSubwindowHeight * scaling_ * 0.74f ) );
    if ( UI::button( "Create Object", { 110 * scaling_, 36 * scaling_ } ) )
    {
        auto itemIt = RibbonSchemaHolder::schema().items.find( "Create Simple Objects");
        if ( itemIt != RibbonSchemaHolder::schema().items.end() )
            itemPressedFn_( itemIt->second.item, true );
        else
            assert( false && "Can not found Create Simple Objects" );
    }

    ImGui::End();
}

void WelcomeWindow::drawCheckbox_()
{
    ImVec2 windowsPos( sceneCenter_.x - ( cSubwindowWidth + cSubwindowSpacing / 2.f ) * scaling_,
        sceneCenter_.y + ( cSubwindowHeight + cSubwindowSpacing / 2.f ) * scaling_ );
    ImVec2 windowsSize( cSubwindowWidth * scaling_, ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2 );
    ImGui::SetNextWindowPos( windowsPos, ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowsSize, ImGuiCond_Always );
    ImGui::PushStyleColor( ImGuiCol_Border, Color::transparent().getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_WindowBg, Color::transparent().getUInt32() );
    if ( !ImGui::Begin( "##CheckboxWindow", &visible_, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_Modal ) )
    {
        ImGui::PopStyleColor( 2 );
        return;
    }
    ImGui::PopStyleColor( 2 );

    UI::checkbox( "Show on Startup", &showOnStartup_ );

    ImGui::End();
}

bool WelcomeWindow::beginSubwindow_( const char* name, Vector2f pos, Vector2f size )
{
    ImGui::SetNextWindowPos( ImVec2( pos.x, pos.y ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( ImVec2( size.x, size.y ), ImGuiCond_Always );
    return ImGui::Begin( name, &visible_, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_Modal );
}

}
