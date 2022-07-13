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
#include "MRShortcutManager.h"
#include <GLFW/glfw3.h>

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif


namespace
{



// goes up and down on scene tree, selecting objects with different modifiers


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
    renameBuffer_ = selected[0]->name();
    showRenameModal_ = true;
}

void Menu::allowObjectsRemoval( bool allow )
{
    allowRemoval_ = allow;
}

void Menu::allowSceneReorder( bool allow )
{
    allowSceneReorder_ = allow;
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













void Menu::setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec )
{
    frameTimeMillisecThreshold_ = maxGoodTimeMillisec;
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
