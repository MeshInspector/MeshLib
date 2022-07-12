#include "MRMenu.h"
#include "MRFileDialog.h"
#include "MRMeshModifier.h"
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRBitSet.h"
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include "MRProgressBar.h"
#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRHistoryStore.h"
#include "ImGuiHelpers.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRToFromEigen.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRCommandLoop.h"
#include "MRRibbonButtonDrawer.h"
#include "MRColorTheme.h"
#include "MRMesh/MRObjectLabel.h"
#include <GLFW/glfw3.h>

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Dense>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace
{

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cEventCounterNames =
{
    "Mouse Down",
    "Mouse Up",
    "Mouse Move",
    "Mouse Scroll",
    "Key Down",
    "Key Up",
    "Key Repeat",
    "Char Pressed"
};

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cGLPrimitivesCounterNames =
{
    "Point Array Size",
    "Line Array Size",
    "Triangle Array Size",
    "Point Elements Number",
    "Line Elements Number",
    "Triangle Elements Number"
};

const ImVec4 undefined = ImVec4(0.5f,0.5f,0.5f,0.5f);

bool objectHasRealChildren( const MR::Object& object )
{
    bool res = false;
    for ( const auto& child : object.children() )
    {
        if ( !child->isAncillary() )
            res = true;
        else
            res = objectHasRealChildren( *child );

        if ( res )
            break;
    }
    return res;
}

// at least one of selected is true - first,
// all selected are true - second
std::pair<bool, bool> getRealValue( const std::vector<std::shared_ptr<MR::VisualObject>>& selected,
                                    unsigned type, MR::ViewportMask viewportId )
{
    bool atLeastOneTrue = false;
    bool allTrue = true;
    for ( const auto& data : selected )
    {
        bool isThisTrue = data && data->getVisualizeProperty( type, viewportId );
        atLeastOneTrue = atLeastOneTrue || isThisTrue;
        allTrue = allTrue && isThisTrue;
    }
    allTrue = allTrue && atLeastOneTrue;
    return {atLeastOneTrue,allTrue};
}

// goes up and down on scene tree, selecting objects with different modifiers
void changeSelection( bool selectNext, int mod )
{
    using namespace MR;
    const auto selectable = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selectNext )
    {
        auto nextIt = std::find_if( selectable.rbegin(), selectable.rend(), [] ( const std::shared_ptr<Object>& obj )
        {
            return obj->isSelected();
        } );

        Object* next{ nullptr };
        if ( nextIt != selectable.rend() )
        {
            auto dist = int( std::distance( nextIt, selectable.rend() ) );
            if ( dist >= 0 && dist < selectable.size() )
                next = selectable[dist].get();
            if ( dist == selectable.size() )
                next = selectable.back().get();
        }

        if ( mod == 0 ) // uncomment if want multy select holding shift
            for ( const auto& data : selected )
                if ( data && data.get() != next )
                    data->select( false );
        if ( next )
            next->select( true );
    }
    else
    {
        auto prevIt = std::find_if( selectable.begin(), selectable.end(), [] ( const std::shared_ptr<Object>& obj )
        {
            return obj->isSelected();
        } );

        Object* prev{ nullptr };
        if ( prevIt != selectable.end() )
        {
            auto dist = int( std::distance( selectable.begin(), prevIt ) ) - 1;
            if ( dist >= 0 && dist < selectable.size() )
                prev = selectable[dist].get();
            if ( dist == -1 )
                prev = selectable.front().get();
        }

        if ( mod == 0 ) // uncomment if want multy select holding shift
            for ( const auto& data : selected )
                if ( data && data.get() != prev )
                    data->select( false );
        if ( prev )
            prev->select( true );
    }
}

}

namespace MR
{

void Menu::init( MR::Viewer *_viewer )
{
    ImGuiMenu::init( _viewer );

    callback_draw_viewer_menu = [&] ()
    {
        // Draw parent menu content
        draw_mr_menu();
    };

    // Draw additional windows
    callback_draw_custom_window = [&] ()
    {
        draw_scene_list();
        draw_helpers();
        draw_custom_plugins();
    };

    setupShortcuts_();
    spdlog::info( "Press F1 to get see shortcut list." );
}

void Menu::load_font(int font_size)
{
#ifdef _WIN32
    if ( viewer->isGLInitialized() )
    {
        ImGuiIO& io = ImGui::GetIO();

        auto fontPath = getMenuFontPath();

        ImVector<ImWchar> ranges;
        ImFontGlyphRangesBuilder builder;
        addMenuFontRanges_( builder );
        builder.BuildRanges( &ranges );

        io.Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), font_size * menu_scaling(),
            nullptr, ranges.Data );
        io.Fonts->Build();
    }
    else
    {
        ImGuiMenu::load_font( font_size );
        ImGui::GetIO().Fonts[0].Build();
    }
#else
    ImGuiMenu::load_font( font_size );
    //TODO: expand for non-Windows systems
#endif
}

void selectRecursive( Object& obj )
{
    obj.select( true );
    for ( auto& child : obj.children() )
        if ( child )
            selectRecursive( *child );
}

void Menu::draw_mr_menu()
{
    // Mesh
    ProgressBar::setup( menu_scaling() );
    const auto& viewportParameters = viewer->viewport().getParameters();
    if ( ImGui::CollapsingHeader( "Main", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        draw_history_block_();
        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( ImGui::Button( "Load##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) )
        {
            auto filenames = openFilesDialog( { {},{},MeshLoad::getFilters() | PointsLoad::Filters | SceneFileFilters } );
            if ( !filenames.empty() )
            {
                SCOPED_HISTORY( "Load files" );
                for ( const auto& filename : filenames )
                {
                    if ( !filename.empty() )
                        viewer->load_file( filename );
                }
                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
            }
        }
        ImGui::SameLine( 0, p );
        draw_open_recent_button_();
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "Load Dir##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto directory = openFolderDialog();
            if ( !directory.empty() )
            {
                auto container = makeObjectTreeFromFolder( directory );
                if ( container.has_value() && !container->children().empty() )
                {
                    auto obj = std::make_shared<Object>( std::move( container.value() ) );
                    obj->setName( utf8string( directory.stem() ) );
                    selectRecursive( *obj );
                    AppendHistory<ChangeSceneAction>( "Load Dir", obj, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( obj );
                    viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                }
#ifndef __EMSCRIPTEN__
                else
                {
                    ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [directory, viewer = viewer]() -> std::function<void()>
                    {
                        ProgressBar::nextTask( "Load DICOM Folder" );
                        auto voxelsObject = VoxelsLoad::loadDCMFolder( directory, 4, ProgressBar::callBackSetProgress );
                        if ( voxelsObject && !ProgressBar::isCanceled() )
                        {
                            auto bins = voxelsObject->histogram().getBins();
                            auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

                            ProgressBar::nextTask( "Create ISO surface" );
                            voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                            voxelsObject->select( true );
                            return [viewer, voxelsObject]()
                            {
                                AppendHistory<ChangeSceneAction>( "Load Voxels", voxelsObject, ChangeSceneAction::Type::AddObject );
                                SceneRoot::get().addChild( voxelsObject );
                                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                            };
                        }
                        return {};
                    }, 2 );
                }
#endif
            }
        }

        if ( ImGui::Button( "Save##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto filters = MeshSave::Filters | LinesSave::Filters | PointsSave::Filters;
#ifndef __EMSCRIPTEN__
            filters = filters | VoxelsSave::Filters;
#endif
            auto savePath = saveFileDialog( { {}, {}, filters } );
            if ( !savePath.empty() )
                viewer->save_mesh_to_file( savePath );                
        }
        ImGui::SameLine( 0, p );

        if ( ImGui::Button( "Save Scene##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto savePath = saveFileDialog( { {},{},SceneFileFilters } );

            ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [savePath, &root = SceneRoot::get(), viewer = this->viewer]()->std::function<void()>
			{
                auto res = serializeObjectTree( root, savePath, []( float progress )
                {
                    return ProgressBar::setProgress( progress );
                } );
                if ( !res.has_value() )
                    spdlog::error( res.error() );

                return[savePath, viewer, success = res.has_value()]()
                {
                    if ( success )
                        viewer->recentFilesStore.storeFile( savePath );
                };
			});
        }

        if ( ImGui::Button( "New Issue##Main", ImVec2( w, 0 ) ) )
        {
            OpenLink( "https://meshinspector.github.io/ReportIssue" );
        }
        if ( ImGui::Button( "Capture Screen##Main", ImVec2( w, 0 ) ) )
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t( now );
            auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", fmt::localtime( t ) );

            auto savePath = saveFileDialog( { name,{},ImageSave::Filters } );
            if ( !savePath.empty() )
            {
                auto image = viewer->captureScreenShot();
                auto res = ImageSave::toAnySupportedFormat( image, savePath );

                if ( !res.has_value() )
                    spdlog::warn( "Error saving screenshot: {}", res.error() );
            }
        }
    }

    // Viewing options
    if ( ImGui::CollapsingHeader( "Viewing Options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::PushItemWidth( 80 * menu_scaling() );
        auto fov = viewportParameters.cameraViewAngle;
        ImGui::DragFloatValid( "Camera FOV", &fov, 0.001f, 0.01f, 179.99f );
        viewer->viewport().setCameraViewAngle( fov );

        bool showGlobalBasis = viewer->globalBasisAxes->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show Global Basis", &showGlobalBasis );
        viewer->viewport().showGlobalBasis( showGlobalBasis );

        bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show rotation center", &showRotCenter );
        viewer->viewport().showRotationCenter( showRotCenter );

        // Orthographic view
        bool orth = viewportParameters.orthographic;
        ImGui::Checkbox( "Orthographic view", &orth );
        viewer->viewport().setOrthographic( orth );
        
        bool flatShading = SceneSettings::get( SceneSettings::Type::MeshFlatShading );
        bool flatShadingBackup = flatShading;
        ImGui::Checkbox( "Default shading flat", &flatShading );
        if ( flatShadingBackup != flatShading )
            SceneSettings::set( SceneSettings::Type::MeshFlatShading, flatShading );
        ImGui::PopItemWidth();

        bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show axes", &showAxes );
        viewer->viewport().showAxes( showAxes );

        const std::string typeColorEditStr = "Background";
        auto backgroundColor = getStoredColor_( typeColorEditStr, viewportParameters.backgroundColor );
        if ( ImGui::ColorEdit4( typeColorEditStr.c_str(), &backgroundColor.x,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
            storedColor_ = { typeColorEditStr,backgroundColor };
        else if ( !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) && storedColor_ && storedColor_->first == typeColorEditStr )
            storedColor_ = {};
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );
    }

    if( ImGui::Button( "Fit Data", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::Visible } );
    }
    if( ImGui::Button( "Fit Selected", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::SelectedPrimitives } );
    }

    if ( viewer->isAlphaSortAvailable() )
    {
        bool alphaSortBackUp = viewer->isAlphaSortEnabled();
        bool alphaBoxVal = alphaSortBackUp;
        ImGui::Checkbox( "Alpha Sort", &alphaBoxVal );
        if ( alphaBoxVal != alphaSortBackUp )
            viewer->enableAlphaSort( alphaBoxVal );
    }

    if ( ImGui::CollapsingHeader( "Viewports" ) )
    {
        auto configBackup = viewportConfig_;
        ImGui::RadioButton( "Single", (int*) &viewportConfig_, ViewportConfigurations::Single );
        ImGui::RadioButton( "Horizontal", (int*) &viewportConfig_, ViewportConfigurations::Horizontal );
        ImGui::RadioButton( "Vertical", (int*) &viewportConfig_, ViewportConfigurations::Vertical );
        ImGui::RadioButton( "Quad", (int*) &viewportConfig_, ViewportConfigurations::Quad );
        if ( configBackup != viewportConfig_ )
        {
            for ( int i = int( viewer->viewport_list.size() ) - 1; i > 0; --i )
                viewer->erase_viewport( i );

            auto win = glfwGetCurrentContext();
            int window_width, window_height;
            glfwGetWindowSize( win, &window_width, &window_height );

            auto bounds = viewer->getViewportsBounds();

            float width = MR::width( bounds );
            float height = MR::height( bounds );

            ViewportRectangle rect;
            switch ( viewportConfig_ )
            {
                case Vertical:
                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height;
                    viewer->viewport().setViewportRect( rect );

                    rect.min.x = bounds.min.x + width * 0.5f;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height;
                    viewer->append_viewport( rect );
                    break;
                case Horizontal:
                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->viewport().setViewportRect( rect );

                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y + height * 0.5f;
                    rect.max.x = rect.min.x + width;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->append_viewport( rect );
                    break;
                case Quad:
                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->viewport().setViewportRect( rect );

                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y + height * 0.5f;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->append_viewport( rect );

                    rect.min.x = bounds.min.x + width * 0.5f;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->append_viewport( rect );

                    rect.min.x = bounds.min.x + width * 0.5f;
                    rect.min.y = bounds.min.y + height * 0.5f;
                    rect.max.x = rect.min.x + width * 0.5f;
                    rect.max.y = rect.min.y + height * 0.5f;
                    viewer->append_viewport( rect );
                    break;
                case Single:
                default:
                    rect.min.x = bounds.min.x;
                    rect.min.y = bounds.min.y;
                    rect.max.x = rect.min.x + width;
                    rect.max.y = rect.min.y + height;
                    viewer->viewport().setViewportRect( rect );
                    break;
            }
            postResize_( window_width, window_height );
        }
    }

    if ( ImGui::CollapsingHeader( "Clipping plane" ) )
    {
        auto plane = viewportParameters.clippingPlane;
        auto showPlane = viewer->clippingPlaneObject->isVisible( viewer->viewport().id );
        plane.n = plane.n.normalized();
        auto w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth( w );
        ImGui::DragFloatValid3( "##ClippingPlaneNormal", &plane.n.x, 1e-3f );
        ImGui::SetNextItemWidth( w / 2.0f );
        ImGui::DragFloatValid( "##ClippingPlaneD", &plane.d, 1e-3f );
        ImGui::SameLine();
        ImGui::Checkbox( "Show##ClippingPlane", &showPlane );
        viewer->viewport().setClippingPlane( plane );
        viewer->viewport().showClippingPlane( showPlane );
    }
    ImGui::Text( "Current view: %d", viewer->viewport().id.value() );
    mainWindowPos_ = ImGui::GetWindowPos();
    mainWindowSize_ = ImGui::GetWindowSize();
}

void Menu::draw_scene_list()
{
    const auto allObj = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    auto selectedObjs = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    // Define next window position + size
    ImGui::SetNextWindowPos( ImVec2( 180 * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 230 * menu_scaling(), 300 * menu_scaling() ), ImGuiCond_FirstUseEver );
    ImGui::Begin(
        "Scene",nullptr
    );
    draw_scene_list_content( selectedObjs, allObj );

    sceneWindowPos_ = ImGui::GetWindowPos();
    sceneWindowSize_ = ImGui::GetWindowSize();
    ImGui::End();

    draw_selection_properties( selectedObjs );
}

void Menu::draw_scene_list_content( const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    // mesh with index 0 is Ancillary, and cannot be removed
    // it can be cleaned but it is inconsistent, so this mesh is untouchable
    int uniqueCounter = 0;
    ImGui::BeginChild( "Meshes", ImVec2( -1, -1 ), true );
    auto children = SceneRoot::get().children();
    for ( const auto& child : children )
        draw_object_recurse_( *child, selected, all, uniqueCounter );
    makeDragDropTarget_( SceneRoot::get(), false, true, uniqueCounter + 1 );
    //ImGui::SetWindowSize( ImVec2(size.x,0) );
    ImGui::EndChild();
    sceneOpenCommands_.clear();

    reorderSceneIfNeeded_();
}

void Menu::draw_custom_plugins()
{    
    pluginsCache_.validate( viewer->plugins );
    StateBasePlugin* enabled = pluginsCache_.findEnabled();

    float availibleWidth = 200.0f * menu_scaling();

    auto selectedObjects = getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    auto selectedVisObjects = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );

    ImGui::SetNextWindowPos( ImVec2( 410.0f * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 0.0f, 0.0f ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Plugins", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize );

    ImGui::SetCursorPosX( 570.f );
    if ( ImGui::InputText( "Search", searchPluginsString_ ) )
    {
        Viewer::instanceRef().incrementForceRedrawFrames( 2, true );
    }

    auto& colors = ImGui::GetStyle().Colors;
    auto backUpButtonColor = colors[ImGuiCol_Button];
    auto backUpTextColor = colors[ImGuiCol_Text];

    ImGui::BeginTabBar( "##CustomPluginsTabs", ImGuiTabBarFlags_TabListPopupButton );

    const int pluginsPerLine = 4;
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        StatePluginTabs tab = StatePluginTabs( t );
        const auto& plugins = pluginsCache_.getTabPlugins( tab );
        int counter = 0;
        for ( auto& plugin : plugins )
        {
            if ( !plugin->checkStringMask( searchPluginsString_ ) )
                continue;

            if ( counter == 0 )
            {
                if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                    break;
            }

            std::string requirements = plugin->isAvailable( selectedObjects );
            bool canEnable = !enabled && requirements.empty();

            if ( plugin->isEnabled() )
            {
                colors[ImGuiCol_Button] = { 0.0f,0.8f,0.0f,1.0f };
                colors[ImGuiCol_Text] = { 0.0f,0.0f,0.0f,1.0f };
            }
            else if ( !canEnable )
            {
                colors[ImGuiCol_Button] = { 0.5f,0.5f,0.5f,1.0f };
                colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            }

            if ( counter % pluginsPerLine != 0 )
                ImGui::SameLine();

            if ( ImGui::Button( plugin->plugin_name.c_str(), ImVec2( availibleWidth, 0 ) ) )
            {
                if ( plugin->isEnabled() )
                    plugin->enable( false );
                else if ( canEnable )
                    plugin->enable( true );
            }
            ++counter;

            colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            const auto strTooltip = plugin->getTooltip();
            if ( ImGui::IsItemHovered() && ( !strTooltip.empty() || !requirements.empty() ) )
            {
                ImVec2 textSize;
                if ( requirements.empty() )
                    textSize = ImGui::CalcTextSize( strTooltip.c_str(), NULL, false, 400.f );
                else
                {
                    if ( strTooltip.empty() )
                        textSize = ImGui::CalcTextSize( requirements.c_str(), NULL, false, 400.f );
                    else
                        textSize = ImGui::CalcTextSize( ( strTooltip + "\n" + requirements ).c_str(), NULL, false, 400.f );
                }
                ImGui::SetNextWindowContentSize( textSize );
                ImGui::BeginTooltip();
                if ( !strTooltip.empty() )
                {
                    ImGui::TextWrapped( "%s", strTooltip.c_str() );
                }
                if ( !requirements.empty() )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
                    ImGui::TextWrapped( "%s", requirements.c_str() );
                    ImGui::PopStyleColor();
                }
                ImGui::EndTooltip();
            }

            colors[ImGuiCol_Button] = backUpButtonColor;
            colors[ImGuiCol_Text] = backUpTextColor;
        }

        int counterModifier = 0;
        if ( !selectedVisObjects.empty() )
        {
            if ( counter != 0 )
                ImGui::Separator();

            for ( const auto& modifier : modifiers_ )
            {
                if ( tab != modifier->getTab() )
                    continue;

                if ( !modifier->checkStringMask( searchPluginsString_ ) )
                    continue;

                if ( counter + counterModifier == 0 )
                {
                    if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                        break;
                }

                if ( counterModifier % pluginsPerLine != 0 )
                    ImGui::SameLine();

                if ( ImGui::Button( modifier->name().c_str(), ImVec2( availibleWidth, 0 ) ) )
                {
                    // Here should be popups
                    if ( modifier->modify( selectedVisObjects ) )
                        spdlog::info( "{}: success", modifier->name() );
                    else
                        spdlog::warn( "{}: failure", modifier->name() );
                }
                ++counterModifier;
            }
        }
        if ( (counter + counterModifier) != 0 )
        {
            ImGui::EndTabItem();
        }
    }

    ImGui::SetWindowSize( ImGui::GetWindowSize() );
    ImGui::EndTabBar();
    ImGui::End();

    if ( enabled && enabled->isEnabled() )
    {
        if ( allowRemoval_ )
            allowRemoval_ = false;
        enabled->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );
        if ( !enabled->dialogIsOpen() )
            enabled->enable( false );
    }
    else
    {
        if ( !allowRemoval_ )
            allowRemoval_ = true;
    }
}

void Menu::draw_helpers()
{
    if ( ImGui::GetIO().KeysDown[GLFW_KEY_F1] )
    {
        const auto style = ImGui::GetStyle();
        const float hotkeysWindowWidth = 300 * menu_scaling();
        size_t numLines = 3;
        if ( shortcutManager_ )
            numLines += shortcutManager_->getShortcutList().size();
        
        const float hotkeysWindowHeight = ( style.WindowPadding.y * 2 + numLines * (ImGui::GetTextLineHeight() + style.ItemSpacing.y ) );

        ImVec2 windowPos = ImGui::GetMousePos();
        windowPos.x = std::min( windowPos.x, Viewer::instanceRef().window_width - hotkeysWindowWidth );
        windowPos.y = std::min( windowPos.y, Viewer::instanceRef().window_height - hotkeysWindowHeight );

        ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always );
        ImGui::SetNextWindowSize( ImVec2( hotkeysWindowWidth, hotkeysWindowHeight ) );
        ImGui::Begin( "HotKeys", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );

        ImFont font = *ImGui::GetFont();
        font.Scale = 1.2f;
        ImGui::PushFont( &font );
        ImGui::Text( "Hot Key List" );
        ImGui::PopFont();
        ImGui::Text( "" );
        ImGui::Text( "F1 - Show this help with hot keys" );
        if ( shortcutManager_ )
        {
            const auto& shortcutsList = shortcutManager_->getShortcutList();
            for ( const auto& [key, name] : shortcutsList )
                ImGui::Text( "%s - %s", ShortcutManager::getKeyString( key ).c_str(), name.c_str() );
        }
        ImGui::End();
    }

    if ( showStatistics_ )
    {
        const auto style = ImGui::GetStyle();
        const float fpsWindowWidth = 300 * menu_scaling();
        int numLines = 4 + int( Viewer::EventType::Count ) + int( Viewer::GLPrimitivesType::Count ); // 4 - for: prev frame time, swapped frames, total frames, fps;
        // TextHeight +1 for button, ItemSpacing +2 for separators
        const float fpsWindowHeight = ( style.WindowPadding.y * 2 + 
                                        ImGui::GetTextLineHeight() * ( numLines + 2 ) + 
                                        style.ItemSpacing.y * ( numLines + 3 ) +
                                        style.FramePadding.y * 4 );
        const float posX = Viewer::instanceRef().window_width - fpsWindowWidth;
        const float posY = Viewer::instanceRef().window_height - fpsWindowHeight;
        ImGui::SetNextWindowPos( ImVec2( posX, posY ), ImGuiCond_FirstUseEver );
        ImGui::SetNextWindowSize( ImVec2( fpsWindowWidth, fpsWindowHeight ) );
        ImGui::Begin( "##FPS", nullptr, ImGuiWindowFlags_AlwaysAutoResize | //ImGuiWindowFlags_NoInputs | 
                      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );
        for ( int i = 0; i<int( Viewer::GLPrimitivesType::Count ); ++i )
            ImGui::Text( "%s: %zu", cGLPrimitivesCounterNames[i], viewer->getLastFrameGLPrimitivesCount( Viewer::GLPrimitivesType( i ) ) );
        ImGui::Separator();
        for ( int i = 0; i<int( Viewer::EventType::Count ); ++i )
            ImGui::Text( "%s: %zu", cEventCounterNames[i], viewer->getEventsCount( Viewer::EventType( i ) ) );
        ImGui::Separator();
        auto prevFrameTime = viewer->getPrevFrameDrawTimeMillisec();
        if ( prevFrameTime > frameTimeMillisecThreshold_ )
            ImGui::TextColored( ImVec4( 1.0f, 0.3f, 0.3f, 1.0f ), "Previous frame time: %lld ms", prevFrameTime );
        else
            ImGui::Text( "Previous frame time: %lld ms", prevFrameTime );
        ImGui::Text( "Total frames: %zu", viewer->getTotalFrames() );
        ImGui::Text( "Swapped frames: %zu", viewer->getSwappedFrames() );
        ImGui::Text( "FPS: %zu", viewer->getFPS() );

        if ( ImGui::Button( "Reset", ImVec2( -1, 0 ) ) )
        {
            viewer->resetAllCounters();
        }
        if ( ImGui::Button( "Print time to console", ImVec2( -1, 0 ) ) )
        {
            printTimingTreeAndStop();
        }
        ImGui::End();
    }

    if ( show_rename_modal_ )
    {
        show_rename_modal_ = false;
        ImGui::OpenPopup( "Rename object" );
    }

    if ( ImGui::BeginModalNoAnimation( "Rename object", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize ) )
    {
        auto obj = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected ).front();
        if ( !obj )
        {
            ImGui::CloseCurrentPopup();
        }
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();
        ImGui::InputText( "Name", renameBuffer, ImGuiInputTextFlags_AutoSelectAll );

        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( ImGui::Button( "Ok", ImVec2( ( w - p ) / 2.f, 0 ) ) || ImGui::GetIO().KeysDownDuration[GLFW_KEY_ENTER] == 0.0f )
        {
            AppendHistory( std::make_shared<ChangeNameAction>( "Rename object", obj ) );
            obj->setName( renameBuffer );
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "Cancel", ImVec2( ( w - p ) / 2.f, 0 ) ) || ImGui::GetIO().KeysDownDuration[GLFW_KEY_ESCAPE] == 0.0f )
        {
            ImGui::CloseCurrentPopup();
        }

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    auto bgBackUp = ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg];

    if ( !storedError_.empty() && !ImGui::IsPopupOpen( " Error##modal" ) )
    {        
        ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 1, 0.125f, 0.125f, bgBackUp.w );
        ImGui::OpenPopup( " Error##modal" );
    }

    if ( ImGui::BeginModalNoAnimation( " Error##modal", nullptr,
                                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize ) )
    {
        ImGui::Text( "%s", storedError_.c_str() );
        
        ImGui::Spacing();
        ImGui::SameLine( ImGui::GetContentRegionAvail().x * 0.5f - 40.0f, ImGui::GetStyle().FramePadding.x );
        if ( ImGui::Button( "Okay", ImVec2( 80.0f, 0 ) ) || ImGui::GetIO().KeysDownDuration[GLFW_KEY_ENTER] == 0.0f ||
           ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) ) )
        {
            storedError_.clear();
            ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg] = bgBackUp;
            ImGui::CloseCurrentPopup();
        }        

        ImGui::EndPopup();        
    }

}

void Menu::setObjectTreeState( const Object* obj, bool open )
{
    if ( obj )
        sceneOpenCommands_[obj] = open;
}

void Menu::tryRenameSelectedObject()
{
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return;
    renameBuffer = selected[0]->name();
    show_rename_modal_ = true;
}

void Menu::allowObjectsRemoval( bool allow )
{
    allowRemoval_ = allow;
}

void Menu::allowSceneReorder( bool allow )
{
    allowSceneReorder_ = allow;
}

void Menu::showErrorModal( const std::string& error )
{
    show_rename_modal_ = false;
    ImGui::CloseCurrentPopup();
    storedError_ = error;
    // this is needed to correctly resize error window
    getViewerInstance().incrementForceRedrawFrames( 2, true );
}

void Menu::draw_selection_properties( std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    if ( !selectedObjs.empty() )
    {
        // Define next window position + size
        ImGui::SetNextWindowPos( ImVec2( sceneWindowPos_.x, sceneWindowPos_.y + sceneWindowSize_.y ) );
        ImGui::SetNextWindowSize( ImVec2( sceneWindowSize_.x, -1 ) );
        ImGui::Begin(
            "Selection Properties", nullptr,
            ImGuiWindowFlags_NoMove
        );
        draw_selection_properties_content( selectedObjs );
        ImGui::End();
    }
}

void Menu::draw_selection_properties_content( std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    drawSelectionInformation_();

    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    bool allHaveVisualisation = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        if ( !obj )
            return false;
        auto objMesh = obj->asType<ObjectMesh>();
        if ( objMesh && objMesh->mesh() )
            return true;
        auto objPoints = obj->asType<ObjectPoints>();
        if ( objPoints && objPoints->pointCloud() )
            return true;
        auto objLines = obj->asType<ObjectLines>();
        if ( objLines && objLines->polyline() )
            return true;
        return false;
    } );

    drawGeneralOptions_( selectedObjs );
    
    if ( allHaveVisualisation && ImGui::CollapsingHeader( "Draw Options" ) )
    {
        drawDrawOptionsCheckboxes_( selectedVisualObjs );
        drawDrawOptionsColors_( selectedVisualObjs );
    }

    draw_custom_selection_properties( selectedObjs );

    drawRemoveButton_( selectedObjs );


    drawTransform_();
}

void Menu::draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& )
{
}

void Menu::draw_custom_tree_object_properties( Object& )
{
}

void Menu::add_modifier( std::shared_ptr<MeshModifier> modifier )
{
    if ( modifier )
        modifiers_.push_back( modifier );
}

bool Menu::onCharPressed_( unsigned int unicode_key, int modifiers )
{
    if ( MR::ImGuiMenu::onCharPressed_( unicode_key, modifiers ) )
        return true;
    return false;
}

bool Menu::onKeyDown_( int key, int modifiers )
{  
    if ( ImGuiMenu::onKeyDown_( key, modifiers ) )
        return true;  
      
    if ( shortcutManager_ )
        return shortcutManager_->processShortcut( { key,modifiers } );

    return false;
}

bool Menu::onKeyRepeat_( int key, int modifiers )
{
    if ( ImGuiMenu::onKeyRepeat_( key, modifiers ) )
        return true;
    if ( shortcutManager_ )
        return shortcutManager_->processShortcut( { key, modifiers } );
    return false;
}

std::filesystem::path Menu::getMenuFontPath() const
{
#ifdef _WIN32
    // get windows font
    wchar_t winDir[MAX_PATH];
    GetWindowsDirectoryW( winDir, MAX_PATH );
    std::filesystem::path winDirPath( winDir );
    winDirPath /= "Fonts";
    winDirPath /= "Consola.ttf";
    return winDirPath;
#else
    return {};
#endif
}

void Menu::addMenuFontRanges_( ImFontGlyphRangesBuilder& builder ) const
{
    builder.AddRanges( ImGui::GetIO().Fonts->GetGlyphRangesCyrillic() );
}

float Menu::drawSelectionInformation_()
{
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );

    auto& style = ImGui::GetStyle();

    float resultHeight = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
    if ( ImGui::CollapsingHeader( "Information", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ScrollbarSize, 12.0f );

        const float smallFramePaddingY = 2.f;
        const float smallItemSpacingY = 2.f;
        const float infoHeight = ImGui::GetTextLineHeight() * 8 +
            style.FramePadding.y * 2 +  smallFramePaddingY * 14 +
            style.ItemSpacing.y * 2 + smallItemSpacingY * 5 ;
        resultHeight += infoHeight + style.ItemSpacing.y;

        ImGui::BeginChild( "SceneInformation", ImVec2( 0, infoHeight ), false, ImGuiWindowFlags_HorizontalScrollbar );
        // Compute total faces/verts in selected objects
        size_t totalFaces = 0;
        size_t totalSelectedFaces = 0;
        size_t totalVerts = 0;
        for ( auto pObj : getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            if ( auto mesh = pObj->mesh() )
            {
                totalFaces += mesh->topology.numValidFaces();
                totalSelectedFaces += pObj->numSelectedFaces();
                totalVerts += mesh->topology.numValidVerts();
            }
        }
        for ( auto pObj : getAllObjectsInTree<ObjectLines>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            if ( auto polyline = pObj->polyline() )
            {
                totalVerts += polyline->topology.numValidVerts();
            }
        }

        size_t totalPoints = 0;
        size_t totalSelectedPoints = 0;
        for ( auto pObj : getAllObjectsInTree<ObjectPoints>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            totalPoints += pObj->numValidPoints();
            totalSelectedPoints += pObj->numSelectedPoints();
        }

        auto drawPrimitivesInfo = [&style]( std::string title, size_t value, size_t selected = 0 )
        {
            if ( value )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
                std::string valueStr;
                std::string labelStr;
                const float width = ( ImGui::CalcItemWidth() - style.ItemInnerSpacing.x * 2.f ) / 3.f;
                if ( selected )
                {
                    valueStr = std::to_string( selected ) + " / ";
                    labelStr = "Selected / ";
                }
                valueStr += std::to_string( value );
                labelStr += title;

                ImGui::SetNextItemWidth( width * 2 + style.ItemInnerSpacing.x );
                ImGui::InputText( ( "##" + labelStr ).c_str(), valueStr, ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_AutoSelectAll );
                ImGui::PopStyleColor();
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::Text( "%s", labelStr.c_str() );
            }
        };

        if ( selectedVisualObjs.size() > 1 )
        {
            drawPrimitivesInfo( "Objects", selectedVisualObjs.size() );
        }
        else if ( auto pObj = getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            auto lastRenameObj = lastRenameObj_.lock();
            if ( lastRenameObj != pObj )
            {
                renameBuffer_ = pObj->name();
                lastRenameObj_ = pObj;
            }

            if ( !ImGui::InputText( "Object Name", renameBuffer_, ImGuiInputTextFlags_AutoSelectAll ) )
            {
                if ( renameBuffer_ == pObj->name() )
                {
                    // clear the pointer to reload the name on next frame (if it was changed from outside)
                    lastRenameObj_.reset();
                }
            }
            if ( ImGui::IsItemDeactivatedAfterEdit() )
            {
                AppendHistory( std::make_shared<ChangeNameAction>( "Rename object", pObj ) );
                pObj->setName( renameBuffer_ );
                lastRenameObj_.reset();
            }
        }
        else
            lastRenameObj_.reset();

        const float itemSpacingY = style.ItemSpacing.y;
        style.ItemSpacing.y = 2.f;
        const float framePaddingY = style.FramePadding.y;
        style.FramePadding.y = 2.f;

        drawPrimitivesInfo( "Faces", totalFaces, totalSelectedFaces );
        drawPrimitivesInfo( "Vertices", totalVerts );
        drawPrimitivesInfo( "Points", totalPoints, totalSelectedPoints );

        const float prevItemSpacingY = totalFaces || totalVerts || totalPoints ? style.ItemSpacing.y : itemSpacingY;
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - prevItemSpacingY + itemSpacingY );

        selectionBbox_ = Box3f{};
        Box3f selectionWorldBox;
        for ( auto pObj : selectedVisualObjs )
        {
            selectionBbox_.include( pObj->getBoundingBox() );
            selectionWorldBox.include( pObj->getWorldBox() );
        }
        if ( selectionBbox_.valid() )
        {
            auto drawVec3 = [&style]( std::string title, Vector3f& value )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
                ImGui::InputFloat3( ( "##" + title ).c_str(), &value.x, "%.3f", ImGuiInputTextFlags_ReadOnly );
                ImGui::PopStyleColor();
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::Text( "%s", title.c_str() );
            };
            drawVec3( "Box min", selectionBbox_.min );
            drawVec3( "Box max", selectionBbox_.max );
            auto bsize = selectionBbox_.size();
            drawVec3( "Box size", bsize );

            if ( selectionWorldBox.valid() )
            {
                auto wbsize = selectionWorldBox.size();
                if ( bsize != wbsize )
                    drawVec3( "World box size", wbsize );
            }
        }

        style.ItemSpacing.y = itemSpacingY;
        style.FramePadding.y = framePaddingY;

        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    return resultHeight;
}

// m = q*r
static void decomposeQR( const Matrix3f & m, Matrix3f & q, Matrix3f & r )
{
    Eigen::HouseholderQR<Eigen::MatrixXf> qr( toEigen( m ) );
    q = fromEigen( Eigen::Matrix3f{ qr.householderQ() } );
    r = fromEigen( Eigen::Matrix3f{ qr.matrixQR() } );
    r.y.x = r.z.x = r.z.y = 0;
}

// m = q*r with all diagonal elements in (r) positive
static void decomposePositiveQR( const Matrix3f & m, Matrix3f & q, Matrix3f & r )
{
    decomposeQR( m, q, r );
    Matrix3f sign;
    for( int i = 0; i < 3; ++i )
    {
        if ( r[i][i] < 0 )
            sign[i][i] = -1;
    }
    q = q * sign;
    r = sign * r;
}

float Menu::drawTransform_()
{
    auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );

    auto& style = ImGui::GetStyle();

    float resultHeight_ = 0.f;
    if ( selected.size() == 1 && !selected[0]->isLocked() )
    {
        resultHeight_ = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
        if ( ImGui::CollapsingHeader( "Transform", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            const float transformHeight = ( ImGui::GetTextLineHeight() + style.FramePadding.y * 2 ) * 3 + style.ItemSpacing.y * 2;
            resultHeight_ += transformHeight + style.ItemSpacing.y;
            ImGui::BeginChild( "SceneTransform", ImVec2( 0, transformHeight ) );
            auto& data = *selected.front();
            auto xf = data.xf();

            Matrix3f q, r;
            decomposePositiveQR( xf.A, q, r );

            auto euler = ( 180 / PI_F ) * q.toEulerAngles();
            Vector3f scale{ r.x.x, r.y.y, r.z.z };

            bool inputDeactivated = false;
            bool inputChanged = false;

            ImGui::PushItemWidth( ( ImGui::GetContentRegionAvail().x - 85 * menu_scaling() - style.ItemInnerSpacing.x * 2 ) / 3.f );
            if ( uniformScale_ )
            {
                float midScale = ( scale.x + scale.y + scale.z ) / 3.0f;
                inputChanged = ImGui::DragFloatValid( "##scaleX", &midScale, midScale * 0.01f, 1e-3f, 1e+6f, "%.3f" );
                if ( inputChanged )
                    scale.x = scale.y = scale.z = midScale;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
            }
            else
            {
                inputChanged = ImGui::DragFloatValid( "##scaleX", &scale.x, scale.x * 0.01f, 1e-3f, 1e+6f, "%.3f" );
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = ImGui::DragFloatValid( "##scaleY", &scale.y, scale.y * 0.01f, 1e-3f, 1e+6f, "%.3f" ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = ImGui::DragFloatValid( "##scaleZ", &scale.z, scale.z * 0.01f, 1e-3f, 1e+6f, "%.3f" ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
            }
            ImGui::SameLine();
            RibbonButtonDrawer::GradientCheckbox( "Uni-scale", &uniformScale_ );
            if ( ImGui::IsItemHovered() )
                ImGui::SetTooltip( "%s", "Selects between uniform scaling or separate scaling along each axis" );
            ImGui::PopItemWidth();

            const char* tooltipsRotation[3] = { 
                "Rotation around Ox-axis, degrees",
                "Rotation around Oy-axis, degrees",
                "Rotation around Oz-axis, degrees"
            };
            ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x - 85 * menu_scaling() );
            auto resultRotation = ImGui::DragFloatValid3( "Rotation XYZ", &euler.x, 0.1f, -360.f, 360.f, "%.1f", 0, &tooltipsRotation );
            inputChanged = inputChanged || resultRotation.valueChanged;
            inputDeactivated = inputDeactivated || resultRotation.itemDeactivatedAfterEdit;
            if ( ImGui::IsItemHovered() )
                ImGui::SetTooltip( "%s", "Sequential intrinsic rotations around Oz, Oy and Ox axes." ); // see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations

            if ( inputChanged )
                xf.A = Matrix3f::rotationFromEuler( ( PI_F / 180 ) * euler ) * Matrix3f::scale( scale );

            const char* tooltipsTranslation[3] = { 
                "Translation along Ox-axis",
                "Translation along Oy-axis",
                "Translation along Oz-axis"
            };
            const auto trSpeed = selectionBbox_.valid() ? 0.003f * selectionBbox_.diagonal() : 0.003f;
            ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x - 85 * menu_scaling() );
            auto resultTranslation = ImGui::DragFloatValid3( "Translation", &xf.b.x, trSpeed, 
                                                             std::numeric_limits<float>::lowest(),
                                                             std::numeric_limits<float>::max(),
                                                             "%.3f", 0, &tooltipsTranslation );
            inputDeactivated = inputDeactivated || resultTranslation.itemDeactivatedAfterEdit;

            if ( xfHistUpdated_ )
                xfHistUpdated_ = !inputDeactivated;

            if ( xf != data.xf() && !xfHistUpdated_ )
            {
                AppendHistory<ChangeXfAction>( "Change XF", selected[0] );
                xfHistUpdated_ = true;
            }
            data.setXf( xf );
            ImGui::EndChild();
        }
    }

    return resultHeight_;
}

void Menu::draw_open_recent_button_()
{
    if ( ImGui::BeginCombo( "##Recently Loaded", "##Recently Loaded", ImGuiComboFlags_NoPreview ) )
    {
        auto filenames = viewer->recentFilesStore.getStoredFiles();
        if ( filenames.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( const auto& file : filenames )
        {
            if ( ImGui::Selectable( utf8string( file ).c_str() ) )
            {
                if ( viewer->load_file( file ) )
                {
                    viewer->fitDataViewport();
                }
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
}

void Menu::draw_history_block_()
{
    auto historyStore = viewer->getGlobalHistoryStore();
    if ( !historyStore )
        return;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];
    const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    auto undos = historyStore->getNActions( 10u, HistoryAction::Type::Undo );
    auto redos = historyStore->getNActions( 10u, HistoryAction::Type::Redo );
    if ( undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( ImGui::Button( "Undo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !undos.empty() )
    {
        historyStore->undo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##UndoStack", "##UndoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( undos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < undos.size(); ++i )
        {
            if ( ImGui::Selectable( ( undos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->undo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() && !undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    else if ( !redos.empty() && undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::Button( "Redo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !redos.empty() )
    {
        historyStore->redo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##RedoStack", "##RedoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( redos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < redos.size(); ++i )
        {
            if ( ImGui::Selectable( ( redos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->redo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
}

void Menu::draw_object_recurse_( Object& object, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all, int& counter )
{
    ++counter;
    std::string counterStr = std::to_string( counter );
    const bool isObjSelectable = !object.isAncillary();

    // has selectable children
    bool hasRealChildren = objectHasRealChildren( object );
    bool isOpen{false};
    if ( ( hasRealChildren || isObjSelectable ) )
    {
        makeDragDropTarget_( object, true, true, counter );
        {
            // Visibility checkbox
            bool isVisible = object.isVisible( viewer->viewport().id );
            RibbonButtonDrawer::GradientCheckbox( ( "##VisibilityCheckbox" + counterStr ).c_str(), &isVisible );
            object.setVisible( isVisible, viewer->viewport().id );
            ImGui::SameLine();
        }
        {
            // custom prefix
            drawCustomObjectPrefixInScene_( object );
        }

        const bool isSelected = object.isSelected();

        auto openCommandIt = sceneOpenCommands_.find( &object );
        if ( openCommandIt != sceneOpenCommands_.end() )
            ImGui::SetNextItemOpen( openCommandIt->second );

        if ( !isSelected )
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Header, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame ).getUInt32() );
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectText ).getUInt32() );
        }

        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

        isOpen = ImGui::TreeNodeEx( ( object.name() + "##" + counterStr ).c_str(),
                                    ( hasRealChildren ? ImGuiTreeNodeFlags_DefaultOpen : 0 ) |
                                    ImGuiTreeNodeFlags_OpenOnArrow |
                                    ImGuiTreeNodeFlags_SpanAvailWidth |
                                    ImGuiTreeNodeFlags_Framed |
                                    ( isSelected ? ImGuiTreeNodeFlags_Selected : 0 ) );

        makeDragDropSource_( selected );
        makeDragDropTarget_( object, false, false, 0 );

        if ( isObjSelectable && ImGui::IsItemHovered() )
        {
            bool pressed = !isSelected && ( ImGui::IsMouseClicked( 0 ) || ImGui::IsMouseClicked( 1 ) );
            bool released = isSelected && !dragTrigger_ && !clickTrigger_ && ImGui::IsMouseReleased( 0 );

            if ( pressed )
                clickTrigger_ = true;
            if ( isSelected && clickTrigger_ && ImGui::IsMouseReleased( 0 ) )
                clickTrigger_ = false;

            if ( pressed || released )
            {
                
                auto newSelection = getPreSelection_( &object, ImGui::GetIO().KeyShift, ImGui::GetIO().KeyCtrl, selected, all );
                if ( ImGui::GetIO().KeyCtrl )
                {
                    for ( auto& sel : newSelection )
                    {
                        const bool select = ImGui::GetIO().KeyShift || !sel->isSelected();
                        sel->select( select );
                        if ( showNewSelectedObjects_ && select )
                                sel->setGlobalVisibilty( true );
                    }
                }
                else
                {
                    for ( const auto& data : selected )
                    {
                        auto inNewSelList = std::find( newSelection.begin(), newSelection.end(), data.get() );
                        if ( inNewSelList == newSelection.end() )
                            data->select( false );
                    }
                    for ( auto& sel : newSelection )
                    {
                        sel->select( true );
                        if ( showNewSelectedObjects_ )
                            sel->setGlobalVisibilty( true );
                    }
                }
            }

        }

        ImGui::PopStyleColor( isSelected ? 2 : 1 );
        ImGui::PopStyleVar();

        if ( isSelected )
            drawSceneContextMenu_( selected );
    }
    if ( isOpen )
    {
        draw_custom_tree_object_properties( object );
        bool infoOpen = false;
        auto lines = object.getInfoLines();
        if ( hasRealChildren && !lines.empty() )
            infoOpen = ImGui::TreeNodeEx( "Info: ", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed );

        if ( infoOpen || !hasRealChildren )
        {
            auto itemSpacing = ImGui::GetStyle().ItemSpacing;
            auto framePadding = ImGui::GetStyle().FramePadding;
            framePadding.y = 2.0f;
            itemSpacing.y = 2.0f;
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
            ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
            for ( const auto& str : lines )
            {
                ImGui::TreeNodeEx( str.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_Framed );
                ImGui::TreePop();
            }
            ImGui::PopStyleVar( 3 );
            ImGui::PopStyleColor();
        }

        if ( hasRealChildren )
        {
            if ( infoOpen )
                ImGui::TreePop();

            auto children = object.children();
            for ( const auto& child : children )
                draw_object_recurse_( *child, selected, all, counter );

            makeDragDropTarget_( object, false, true, 0 );
        }
        ImGui::TreePop();
    }
}

std::vector<Object*> Menu::getPreSelection_( Object* meshclicked,
                                             bool isShift, bool isCtrl,
                                             const std::vector<std::shared_ptr<Object>>& selected,
                                             const std::vector<std::shared_ptr<Object>>& all_objects )
{
    if ( selected.empty() || !isShift )
        return {meshclicked};

    const auto& first = isCtrl ? selected.back().get() : selected.front().get();

    auto firstIt = std::find_if( all_objects.begin(), all_objects.end(), [first]( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == first;
    } );
    auto clickedIt = std::find_if( all_objects.begin(), all_objects.end(), [meshclicked]( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == meshclicked;
    } );

    size_t start{0};
    std::vector<Object*> res;
    if ( firstIt < clickedIt )
    {
        start = std::distance( all_objects.begin(), firstIt );
        res.resize( std::distance( firstIt, clickedIt + 1 ) );
    }
    else
    {
        start = std::distance( all_objects.begin(), clickedIt );
        res.resize( std::distance( clickedIt, firstIt + 1 ) );
    }
    for ( int i = 0; i < res.size(); ++i )
    {
        res[i] = all_objects[start + i].get();
    }
    return res;
}

bool Menu::make_checkbox( const char* label, bool& checked, bool mixed )
{
    auto backUpCheckColor = ImGui::GetStyle().Colors[ImGuiCol_CheckMark];
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( mixed )
    {
        ImGui::GetStyle().Colors[ImGuiCol_CheckMark] = undefined;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const bool res = RibbonButtonDrawer::GradientCheckbox( label, &checked );
    ImGui::GetStyle().Colors[ImGuiCol_CheckMark] = backUpCheckColor;
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    return res;
}

bool Menu::make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, unsigned type, MR::ViewportMask viewportid )
{
    auto realRes = getRealValue( selectedVisualObjs, type, viewportid );
    bool checked = realRes.first;
    const bool res = make_checkbox( label, checked, !realRes.second && realRes.first );
    if ( checked != realRes.first )
        for ( const auto& data : selectedVisualObjs )
            if ( data )
                data->setVisualizeProperty( checked, type, viewportid );

    return res;
}

template<typename ObjectT>
void Menu::make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label, 
                                std::function<Vector4f( const ObjectT* )> getter, 
                                std::function<void( ObjectT*, const Vector4f& )> setter )
{
    auto color = getter( selectedVisualObjs[0].get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != color )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        color = Vector4f::diagonal( 0.0f ); color[3] = 1.0f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }

    std::string storedName = label;
    for ( const auto& obj : selectedVisualObjs )
        storedName += std::to_string( intptr_t( obj.get() ) );

    const auto colorConstForComparation = color;
    color = getStoredColor_( storedName, Color( color ) );
    ImGui::PushItemWidth( 40 * menu_scaling() );
    if ( ImGui::ColorEdit4( label, &color.x,
        ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
        storedColor_ = { storedName,color };
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( color != colorConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), color );
}

void Menu::make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, std::function<float( const ObjectLinesHolder* )> getter, std::function<void( ObjectLinesHolder*, const float& )> setter )
{
    auto objLines = selectedVisualObjs[0]->asType<ObjectLinesHolder>();
    auto value = getter( objLines );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i]->asType<ObjectLinesHolder>() ) != value )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 0.f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 40 * menu_scaling() );
    ImGui::DragFloatValid( label, &value, 0.02f, 1.f, 10.f, "%.1f" );
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data->asType<ObjectLinesHolder>(), value );
}

bool Menu::drawRemoveButton_( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

    if ( !allowRemoval_ )
    {
        const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    bool clicked = allowRemoval_ ?
        RibbonButtonDrawer::GradientButton( "Remove", ImVec2( -1, 0 ) ) :
        ImGui::Button( "Remove", ImVec2( -1, 0 ) );
    if ( clicked )
    {
        someChanges |= true;
        if ( allowRemoval_ )
        {
            SCOPED_HISTORY( "Remove objects" );
            for ( int i = ( int )selectedObjs.size() - 1; i >= 0; --i )
                if ( selectedObjs[i] )
                {
                    // for now do it by one object
                    AppendHistory<ChangeSceneAction>( "Remove object", selectedObjs[i], ChangeSceneAction::Type::RemoveObject );
                    selectedObjs[i]->detachFromParent();
                }
        }
    }
    if ( !allowRemoval_ )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }

    return someChanges;
}

bool Menu::drawDrawOptionsCheckboxes_( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs )
{
    bool someChanges = false;
    if ( selectedVisualObjs.empty() )
        return someChanges;

    bool allIsObjMesh = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectMeshHolder>();
    } );
    bool allIsObjLines = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), []( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectLinesHolder>();
    } );
    bool allIsObjPoints = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectPointsHolder>();
    } );
    bool allIsObjLabels = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectLabel>();
    } );

    const auto& viewportid = viewer->viewport().id;

    if ( allIsObjMesh )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Flat Shading", MeshVisualizePropertyType::FlatShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Edges", MeshVisualizePropertyType::Edges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Edges", MeshVisualizePropertyType::SelectedEdges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Faces", MeshVisualizePropertyType::SelectedFaces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Borders", MeshVisualizePropertyType::BordersHighlight, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Faces", MeshVisualizePropertyType::Faces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Only Odd Fragments", MeshVisualizePropertyType::OnlyOddFragments, viewportid );
    }
    if ( allIsObjLines )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Points", LinesVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Smooth corners", LinesVisualizePropertyType::Smooth, viewportid );
        make_width( selectedVisualObjs, "Line width", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getLineWidth();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setLineWidth( value );
        } );
        make_width( selectedVisualObjs, "Point size", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getPointSize();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setPointSize( value );
        } );
    }
    if ( allIsObjPoints )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Points", PointsVisualizePropertyType::SelectedVertices, viewportid );
    }
    if ( allIsObjLabels )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Always on top", VisualizeMaskType::DepthTest, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Invert Normals", VisualizeMaskType::InvertedNormals, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Name", VisualizeMaskType::Name, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Labels", VisualizeMaskType::Labels, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Clipping", VisualizeMaskType::ClippedByPlane, viewportid );

    return someChanges;
}

bool Menu::drawDrawOptionsColors_( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs )
{
    bool someChanges = false;
    const auto selectedMeshObjs = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    const auto selectedPointsObjs = getAllObjectsInTree<ObjectPointsHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selectedVisualObjs.empty() )
        return someChanges;

    make_color_selector<VisualObject>( selectedVisualObjs, "Selected color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor() );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ) );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Unselected color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor( false ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), false );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Back Faces color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getBackColor() );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setBackColor( Color( color ) );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Labels color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getLabelsColor() );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setLabelsColor( Color( color ) );
    } );

    if ( !selectedMeshObjs.empty() )
    {
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getEdgesColor() );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setEdgesColor( Color( color ) );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Faces color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedFacesColor() );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedFacesColor( Color( color ) );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedEdgesColor() );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedEdgesColor( Color( color ) );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Borders color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getBordersColor() );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setBordersColor( Color( color ) );
        } );
    }
    if ( !selectedPointsObjs.empty() )
    {
        make_color_selector<ObjectPointsHolder>( selectedPointsObjs, "Selected Points color", [&]( const ObjectPointsHolder* data )
        {
            return Vector4f( data->getSelectedVerticesColor() );
        }, [&]( ObjectPointsHolder* data, const Vector4f& color )
        {
            data->setSelectedVerticesColor( Color( color ) );
        } );
    }


    return someChanges;
}

bool Menu::drawGeneralOptions_( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( !selectedVisualObjs.empty() )
    {
        const auto& viewportid = viewer->viewport().id;
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Visibility", VisualizeMaskType::Visibility, viewportid );
    }

    bool hasLocked = false, hasUnlocked = false;
    for ( const auto& s : selectedObjs )
    {
        if ( s->isLocked() )
            hasLocked = true;
        else
            hasUnlocked = true;
    }
    const bool mixedLocking = hasLocked && hasUnlocked;
    bool checked = hasLocked;
    someChanges |= make_checkbox( "Lock Transform", checked, mixedLocking );
    if ( checked != hasLocked )
        for ( const auto& s : selectedObjs )
            s->setLocked( checked );

    return someChanges;
}

void Menu::setupShortcuts_()
{
    if ( !shortcutManager_ )
        shortcutManager_ = std::make_shared<ShortcutManager>();

    shortcutManager_->setShortcut( { GLFW_KEY_H,0 }, { "Toggle selected objects visibility",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
        bool atLeastOne = false;
        for ( const auto& data : selected )
            if ( data )
                if ( data->isVisible( viewportid ) )
                {
                    atLeastOne = true;
                    break;
                }
        for ( const auto& data : selected )
            if ( data )
                data->setVisible( !atLeastOne, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_D,0 }, { "Toggle statistics window",[this] ()
    {
        showStatistics_ = !showStatistics_;
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_F,0 }, { "Toggle shading of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::FlatShading, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_I,0 }, { "Invert normals of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( VisualizeMaskType::InvertedNormals, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_L,0 }, { "Toggle edges on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
                sel->toggleVisualizeProperty( MeshVisualizePropertyType::Edges, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_O,0 }, { "Toggle orthographic in current viewport",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        viewport.setOrthographic( !viewport.getParameters().orthographic );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_T,0 }, { "Toggle faces on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::Faces, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,0 }, { "Select next object",[] ()
    {
        changeSelection( true,0 );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,GLFW_MOD_SHIFT }, { "Add next object to selection",[] ()
    {
        changeSelection( true,GLFW_MOD_SHIFT );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,0 }, { "Select previous object",[] ()
    {
        changeSelection( false,0 );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,GLFW_MOD_SHIFT }, { "Add previous object to selection",[] ()
    {
        changeSelection( false,GLFW_MOD_SHIFT );
    } } );
}

void Menu::setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec )
{
    frameTimeMillisecThreshold_ = maxGoodTimeMillisec;
}

void Menu::reorderSceneIfNeeded_()
{
    if ( !allowSceneReorder_ )
        return;

    const bool filledReorderCommand = !sceneReorderCommand_.who.empty() && sceneReorderCommand_.to;
    const bool sourceNotTarget = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), [target = sceneReorderCommand_.to]( auto it )
    {
        return it != target;
    } );
    const bool trueTarget = !sceneReorderCommand_.before || sceneReorderCommand_.to->parent();
    const bool trueSource = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), []( auto it )
    {
        return bool( it->parent() );
    } );
    if ( !( filledReorderCommand && sourceNotTarget && trueSource && trueTarget ) )
    {
        sceneReorderCommand_ = {};
        return;
    }

    bool dragOrDropFailed = false;
    std::shared_ptr<Object> childTo = nullptr;
    if ( sceneReorderCommand_.before )
    {
        for ( auto childToItem : sceneReorderCommand_.to->parent()->children() )
            if ( childToItem.get() == sceneReorderCommand_.to )
            {
                childTo = childToItem;
                break;
            }
        assert( childTo );
    }

    struct MoveAction
    {
        std::shared_ptr<ChangeSceneAction> detachAction;
        std::shared_ptr<ChangeSceneAction> attachAction;
    };
    std::vector<MoveAction> actionList;
    for ( const auto& source : sceneReorderCommand_.who )
    {
        std::shared_ptr<Object> sourcePtr = nullptr;
        for ( auto child : source->parent()->children() )
            if ( child.get() == source )
            {
                sourcePtr = child;
                break;
            }
        assert( sourcePtr );

        auto detachAction = std::make_shared<ChangeSceneAction>( "Detach object", sourcePtr, ChangeSceneAction::Type::RemoveObject );
        bool detachSuccess = sourcePtr->detachFromParent();
        if ( !detachSuccess )
        {
            showErrorModal( "Cannot preform such reorder" );
            dragOrDropFailed = true;
            break;
        }

        auto attachAction = std::make_shared<ChangeSceneAction>( "Attach object", sourcePtr, ChangeSceneAction::Type::AddObject );
        bool attachSucess{ false };
        if ( !sceneReorderCommand_.before )
            attachSucess = sceneReorderCommand_.to->addChild( sourcePtr );
        else
            attachSucess = sceneReorderCommand_.to->parent()->addChildBefore( sourcePtr, childTo );
        if ( !attachSucess )
        {
            detachAction->action( HistoryAction::Type::Undo );
            showErrorModal( "Cannot preform such reorder" );
            dragOrDropFailed = true;
            break;
        }

        actionList.push_back( { detachAction, attachAction } );
    }

    if ( dragOrDropFailed )
    {
        for ( int i = int( actionList.size() ) - 1; i >= 0; --i )
        {
            actionList[i].attachAction->action( HistoryAction::Type::Undo );
            actionList[i].detachAction->action( HistoryAction::Type::Undo );
        }
    }
    else
    {
        SCOPED_HISTORY( "Reorder scene" );
        for ( const auto& moveAction : actionList )
        {
            AppendHistory( moveAction.detachAction );
            AppendHistory( moveAction.attachAction );
        }
    }
    sceneReorderCommand_ = {};
    dragTrigger_ = false;
}

Vector4f Menu::getStoredColor_( const std::string& str, const Color& defaultColor ) const
{
    if ( !storedColor_ || storedColor_->first != str )
        return Vector4f( defaultColor );
    return storedColor_->second;
}

void Menu::makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload )
{
    if ( !allowSceneReorder_ || payload.empty() )
        return;
    if ( ImGui::BeginDragDropSource( ImGuiDragDropFlags_AcceptNoDrawDefaultRect ) )
    {
        dragTrigger_ = true;

        std::vector<Object*> vectorObjPtr;
        for ( auto& ptr : payload )
            vectorObjPtr.push_back( ptr.get() );

        ImGui::SetDragDropPayload( "_TREENODE", vectorObjPtr.data(), sizeof( Object* ) * vectorObjPtr.size() );
        std::string allNames;
        allNames = payload[0]->name();
        for ( int i = 1; i < payload.size(); ++i )
            allNames += "\n" + payload[i]->name();
        ImGui::Text( "%s", allNames.c_str() );
        ImGui::EndDragDropSource();
    }

}

void Menu::makeDragDropTarget_( Object& target, bool before, bool betweenLine, int counter )
{
    if ( !allowSceneReorder_ )
        return;
    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    ImVec2 curPos{};
    bool lineDrawed = false;
    if ( payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE" && betweenLine )
    {
        lineDrawed = true;
        curPos = ImGui::GetCursorPos(); 
        auto width = ImGui::GetContentRegionAvail().x;
        ImGui::ColorButton( ( "##InternalDragDropArea" + std::to_string( counter ) ).c_str(),
            ImVec4(0,0,0,0),
            0, ImVec2( width, 4.0f ) );
    }
    if ( ImGui::BeginDragDropTarget() )
    {
        if ( lineDrawed )
        {
            ImGui::SetCursorPos( curPos );
            auto width = ImGui::GetContentRegionAvail().x;
            ImGui::ColorButton( ( "##ColoredInternalDragDropArea" + std::to_string( counter ) ).c_str(),
                ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered],
                0, ImVec2( width, 4.0f ) );
        }
        if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload( "_TREENODE" ) )
        {
            assert( payload->DataSize % sizeof( Object* ) == 0 );
            Object** objArray = ( Object** )payload->Data;
            const int size = payload->DataSize / sizeof( Object* );
            std::vector<Object*> vectorObj(size);
            for ( int i = 0; i < size; ++i )
                vectorObj[i] = objArray[i];
            sceneReorderCommand_ = { vectorObj, &target, before };
        }
        ImGui::EndDragDropTarget();
    }
}


void Menu::PluginsCache::validate( const std::vector<ViewerPlugin*>& viewerPlugins )
{
    // if same then cache is valid
    if ( viewerPlugins == allPlugins_ )
        return;

    allPlugins_ = viewerPlugins;

    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
        sortedCustomPlufins_[t] = {};
    for ( const auto& plugin : allPlugins_ )
    {
        StateBasePlugin * customPlugin = dynamic_cast< StateBasePlugin* >( plugin );
        if ( customPlugin )
            sortedCustomPlufins_[int( customPlugin->getTab() )].push_back( customPlugin );
    }
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        auto& tabPlugins = sortedCustomPlufins_[t];
        std::sort( tabPlugins.begin(), tabPlugins.end(), [] ( const auto& a, const auto& b )
        {
            return a->sortString() < b->sortString();
        } );
    }
}

StateBasePlugin* Menu::PluginsCache::findEnabled() const
{
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        const auto& tabPlugins = sortedCustomPlufins_[t];
        for ( auto plug : tabPlugins )
            if ( plug->isEnabled() )
                return plug;
    }
    return nullptr;
}

const std::vector<StateBasePlugin*>& Menu::PluginsCache::getTabPlugins( StatePluginTabs tab ) const
{
    return sortedCustomPlufins_[int( tab )];
}

}
