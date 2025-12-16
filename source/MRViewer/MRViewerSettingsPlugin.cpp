#include "MRViewerSettingsPlugin.h"
#include "MRRibbonMenu.h"
#include "ImGuiHelpers.h"
#include "MRColorTheme.h"
#include "MRMouseController.h"
#include "MRViewport.h"
#include "MRFileDialog.h"
#include "MRModalDialog.h"
#include "MRCommandLoop.h"
#include "MRViewerSettingsManager.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRRibbonConstants.h"
#include "MRViewer.h"
#include "MRImGuiVectorOperators.h"
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRUIStyle.h"
#include "MRUnitSettings.h"
#include "MRShowModal.h"
#include "MRRibbonSceneObjectsListDrawer.h"
#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRDirectory.h"
#include <MRMesh/MRSceneRoot.h>
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRConfig.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewportGlobalBasis.h"
#include "MRImGuiMultiViewport.h"
#include "MRShortcutManager.h"
#include "MRViewerConfigConstants.h"

namespace
{
const char* getViewerSettingTabName( MR::ViewerSettingsPlugin::TabType tab )
{
    constexpr std::array<const char*, size_t( MR::ViewerSettingsPlugin::TabType::Count )> tabNames{
        "Quick",
        "Application",
        "Control",
        "3D View",
        "Units",
        "Features",
    };
    return tabNames[int( tab )];
}
}

namespace MR
{

ViewerSettingsPlugin::ViewerSettingsPlugin() :
    StatePlugin( "Viewer settings" )
{
    shadowGl_ = std::make_unique<ShadowsGL>();
    CommandLoop::appendCommand( [&] ()
    {
        auto& viewer = getViewerInstance();

        if ( viewer.isGLInitialized() && loadGL() )
        {
            GL_EXEC( glGetIntegerv( GL_MAX_SAMPLES, &maxSamples_ ) );
            storedSamples_ = viewer.getMSAA();
            maxSamples_ = std::max( std::min( maxSamples_, 16 ), storedSamples_ ); // there are some known issues with 32 MSAA
            gpuOverridesMSAA_ = storedSamples_ != viewer.getRequestedMSAA(); // if it fails on application start - gpu overrides settings
#ifdef __EMSCRIPTEN__
            if ( !viewer.isSceneTextureEnabled() )
                maxSamples_ = std::min( maxSamples_, 4 ); // web does not allow more then x4 msaa for main framebuffer
#endif
        }
    }, CommandLoop::StartPosition::AfterWindowAppear );
}

void ViewerSettingsPlugin::drawDialog( ImGuiContext* )
{
    auto menuWidth = 400.0f * UI::scale();

    ImVec2 position = ImGuiMV::Window2ScreenSpaceImVec2( ImVec2( ( viewer->framebufferSize.x - menuWidth ) / 2, viewer->framebufferSize.y / 6.0f ) );
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .position = &position } ) )
        return;

    if ( UI::beginTabBar( "##MainTabs" ) )
    {
        for ( int i = 0; i<int( TabType::Count ); ++i )
        {
            if ( i == int( TabType::Features ) && !viewer->experimentalFeatures )
            {
                // Check if features are considered experimental
                const auto& tabs = RibbonSchemaHolder::schema().tabsOrder;
                auto itFeatures = std::find_if( tabs.begin(), tabs.end(),
                    [] ( const RibbonTab& tab ) { return tab.name == "Features"; } );
                if ( itFeatures != tabs.end() && itFeatures->experimental )
                    continue;
            }
            auto tab = TabType( i );
            bool neetToSelect = orderedTab_ == tab;
            if ( UI::beginTabItem( getViewerSettingTabName( tab ), nullptr, neetToSelect ? ImGuiTabItemFlags_SetSelected : 0 ) )
            {
                if ( neetToSelect )
                    orderedTab_ = TabType::Count;
                activeTab_ = tab;
                drawTab_( menuWidth );
                drawCustomSettings_( "Tools", true );
                UI::endTabItem();
            }
        }
        UI::endTabBar();
    }

    ImGui::EndCustomStatePlugin();
}

void ViewerSettingsPlugin::updateThemes()
{
    selectedUserPreset_ = -1;
    userThemesPresets_.clear();
    userThemesPresets_.push_back( "Dark" );
    userThemesPresets_.push_back( "Light" );
    auto colorThemeType = ColorTheme::getThemeType();
    auto colorThemeName = ColorTheme::getThemeName();
    if ( colorThemeType == ColorTheme::Type::Default )
    {
        if ( colorThemeName == ColorTheme::getPresetName( ColorTheme::Preset::Light ) )
            selectedUserPreset_ = 1;
        else
            selectedUserPreset_ = 0;
    }

    auto userThemesDir = ColorTheme::getUserThemesDirectory();
    int i = int( userThemesPresets_.size() );
    std::error_code ec;
    if ( std::filesystem::is_directory( userThemesDir, ec ) )
    {
        for ( auto entry : Directory{ userThemesDir, ec } )
        {
            if ( entry.is_regular_file( ec ) )
            {
                auto ext = entry.path().extension().u8string();
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext != u8".json" )
                    break;
                std::string themeName = utf8string( entry.path().stem() );
                userThemesPresets_.push_back( themeName );
                if ( selectedUserPreset_ == -1 && themeName == ColorTheme::getThemeName() )
                    selectedUserPreset_ = i;
                ++i;
            }
        }
    }
}

void ViewerSettingsPlugin::addComboSettings( const TabType tab, std::shared_ptr<ExternalSettings> settings )
{
    comboSettings_[size_t( tab )].push_back( settings );
}

void ViewerSettingsPlugin::delComboSettings( const TabType tab, const ExternalSettings * settings )
{
    [[maybe_unused]] auto c = std::erase_if( comboSettings_[size_t( tab )], [settings]( const auto & v ) { return v.get() == settings; } );
    assert( c == 1 );
}

ViewerSettingsPlugin* ViewerSettingsPlugin::instance()
{
    static ViewerSettingsPlugin* self = [&]()->ViewerSettingsPlugin*
    {
      auto viewerSettingsIt = RibbonSchemaHolder::schema().items.find( "Viewer settings" );
      if ( viewerSettingsIt == RibbonSchemaHolder::schema().items.end() )
          return nullptr;
      return dynamic_cast< ViewerSettingsPlugin* >( viewerSettingsIt->second.item.get() );
      }();
    return self;
}

void ViewerSettingsPlugin::setActiveTab( TabType tab )
{
    orderedTab_ = tab;
}

bool ViewerSettingsPlugin::onEnable_()
{
    backgroundColor_.w = -1.0f;

    updateDialog_();

    return true;
}

bool ViewerSettingsPlugin::onDisable_()
{
    if ( viewer )
        if ( const auto& mgr = viewer->getViewerSettingsManager() )
            mgr->saveSettings( *viewer );
    Config::instance().writeToFile();
    userThemesPresets_.clear();
    return true;
}

void ViewerSettingsPlugin::drawTab_( float menuWidth )
{
    switch ( activeTab_ )
    {
    case MR::ViewerSettingsPlugin::TabType::Quick:
        drawQuickTab_( menuWidth );
        break;
    case MR::ViewerSettingsPlugin::TabType::Application:
        drawApplicationTab_( menuWidth );
        break;
    case MR::ViewerSettingsPlugin::TabType::Control:
        drawControlTab_( menuWidth );
        break;
    case MR::ViewerSettingsPlugin::TabType::Viewport:
        drawViewportTab_( menuWidth );
        break;
    case MR::ViewerSettingsPlugin::TabType::MeasurementUnits:
        drawMeasurementUnitsTab_();
        break;
    case MR::ViewerSettingsPlugin::TabType::Features:
        drawFeaturesTab_();
        break;
    case MR::ViewerSettingsPlugin::TabType::Count:
    default:
        break;
    }
}

void ViewerSettingsPlugin::drawQuickTab_( float menuWidth )
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    if ( !ribbonMenu )
        return;

    drawSeparator_( "General" );

    drawThemeSelector_();

    const auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 2 } );

    drawShadingModeCombo_( false, 200.0f * UI::scale() );
    drawBackgroundButton_( true );

    ImGui::PopStyleVar();

    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;
    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu )
        ribbonMenu->openToolbarCustomize();
    ImGui::SameLine();
    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu )
        ribbonMenu->setShowShortcuts( true );

    drawMouseSceneControlsSettings_( menuWidth );
}

void ViewerSettingsPlugin::drawGlobalSettings_( float buttonWidth )
{
    drawSeparator_( "Global" );

    bool resetClicked = UI::button( "Reset Settings", Vector2f( buttonWidth, 0 ) );
    drawResetDialog_( resetClicked );
}

void ViewerSettingsPlugin::drawApplicationTab_( float menuWidth )
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    if ( !ribbonMenu )
        return;
    const float btnHalfSizeX = 168.0f * UI::scale();

    drawSeparator_( "Interface" );

    const auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
    drawThemeSelector_();

    ImGui::SetNextItemWidth( 200.0f * UI::scale() );
    UI::drag<RatioUnit>( "UI Scale", tempUserScaling_, 0.01f, 0.5f, 4.0f );
    if ( ImGui::IsItemDeactivatedAfterEdit() )
    {
        viewer->getMenuPlugin()->setUserScaling( tempUserScaling_ );
        tempUserScaling_ = viewer->getMenuPlugin()->getUserScaling();
    }

    bool savedDialogsBackUp = viewer->getMenuPlugin()->isSavedDialogPositionsEnabled();
    bool savedDialogsVal = savedDialogsBackUp;
    UI::checkbox( "Save Tool Window Positions", &savedDialogsVal );
    UI::setTooltipIfHovered( "If checked then enables using of saved positions of tool windows in the config file" );
    ImGui::PopStyleVar();

    if ( savedDialogsVal != savedDialogsBackUp )
        viewer->getMenuPlugin()->enableSavedDialogPositions( savedDialogsVal );

    if ( viewer->isMultiViewportAvailable() )
    {
        auto& config = Config::instance();
        bool value = true;
        if ( config.hasBool( cDefaultMultiViewport ) )
            value = config.getBool( cDefaultMultiViewport, true );
        if ( UI::checkbox( "Enable multi-windows", &value ) )
            config.setBool( cDefaultMultiViewport, value );
        UI::setTooltipIfHovered( "Allow tool windows to be moved outside the main window. To apply the changes, need to restart the application." );
    }

    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu )
        ribbonMenu->openToolbarCustomize();

    drawSeparator_( "Behavior" );

    ImGui::SetNextItemWidth( menuWidth * 0.5f );
    if ( ribbonMenu )
    {
        auto sceneObjectsListDrawer = ribbonMenu->getSceneObjectsList();
        if ( sceneObjectsListDrawer )
        {
            UI::checkbox( "Make Visible on Select",
                                                    std::bind( &SceneObjectsListDrawer::getShowNewSelectedObjects, sceneObjectsListDrawer ),
                                                    std::bind( &SceneObjectsListDrawer::setShowNewSelectedObjects, sceneObjectsListDrawer, std::placeholders::_1 ) );
            UI::checkbox( "Deselect on Hide",
                                                    std::bind( &SceneObjectsListDrawer::getDeselectNewHiddenObjects, sceneObjectsListDrawer ),
                                                    std::bind( &SceneObjectsListDrawer::setDeselectNewHiddenObjects, sceneObjectsListDrawer, std::placeholders::_1 ) );

            if ( auto ribbonSceneObjectsListDrawer = std::dynamic_pointer_cast< RibbonSceneObjectsListDrawer >( sceneObjectsListDrawer ) )
            {
                UI::checkbox( "Close Context Menu on Click",
                                                        std::bind( &RibbonSceneObjectsListDrawer::getCloseContextOnChange, ribbonSceneObjectsListDrawer ),
                                                        std::bind( &RibbonSceneObjectsListDrawer::setCloseContextOnChange, ribbonSceneObjectsListDrawer, std::placeholders::_1 ) );
                UI::setTooltipIfHovered( "Close scene context menu on any change or click outside" );
            }
        }

        UI::checkbox( "Auto Close Previous Tool",
                                                std::bind( &RibbonMenu::getAutoCloseBlockingPlugins, ribbonMenu ),
                                                std::bind( &RibbonMenu::setAutoCloseBlockingPlugins, ribbonMenu, std::placeholders::_1 ) );
        UI::setTooltipIfHovered( "Automatically close blocking tool when another blocking tool is activated" );

        UI::checkbox( "Sort Dropped Files",
                                                std::bind( &Viewer::getSortDroppedFiles, viewer ),
                                                std::bind( &Viewer::setSortDroppedFiles, viewer, std::placeholders::_1 ) );
        UI::setTooltipIfHovered( "Whether to sort the filenames received from Drag&Drop in lexicographical order before adding them in scene" );

        UI::checkbox( "Show Experimental Features", &viewer->experimentalFeatures );
        UI::setTooltipIfHovered( "Show experimental or diagnostic tools and controls" );
    }

    drawGlobalSettings_( btnHalfSizeX );

    if ( ribbonMenu )
    {
        drawSeparator_( "Notifications" );

        UI::checkbox( "Time Reports", [&] ()
        {
            return bool( ribbonMenu->getRibbonNotifier().allowedTagMask & NotificationTags::Report );
        }, [&] ( bool on )
        {
            if ( on )
                ribbonMenu->getRibbonNotifier().allowedTagMask |= NotificationTags::Report;
            else
                ribbonMenu->getRibbonNotifier().allowedTagMask &= ~NotificationTags::Report;
        } );
        UI::setTooltipIfHovered( "Show duration of last operation of the application." );

        UI::checkbox( "Recommendations", [&] ()
        {
            return bool( ribbonMenu->getRibbonNotifier().allowedTagMask & NotificationTags::Recommendation );
        }, [&] ( bool on )
        {
            if ( on )
                ribbonMenu->getRibbonNotifier().allowedTagMask |= NotificationTags::Recommendation;
            else
                ribbonMenu->getRibbonNotifier().allowedTagMask &= ~NotificationTags::Recommendation;
        } );
        UI::setTooltipIfHovered( "Show notifications with recommended actions." );

        UI::checkbox( "Implicit Changes", [&] ()
        {
            return bool( ribbonMenu->getRibbonNotifier().allowedTagMask & NotificationTags::ImplicitChanges );
        }, [&] ( bool on )
        {
            if ( on )
                ribbonMenu->getRibbonNotifier().allowedTagMask |= NotificationTags::ImplicitChanges;
            else
                ribbonMenu->getRibbonNotifier().allowedTagMask &= ~NotificationTags::ImplicitChanges;
        } );
        UI::setTooltipIfHovered( "Notify when some changes were made implicitly by the application. (mostly appear on import of non-manifold models)" );

        UI::checkbox( "Important", [&] ()
        {
            return bool( ribbonMenu->getRibbonNotifier().allowedTagMask & NotificationTags::Important );
        }, [&] ( bool on )
        {
            if ( on )
                ribbonMenu->getRibbonNotifier().allowedTagMask |= NotificationTags::Important;
            else
                ribbonMenu->getRibbonNotifier().allowedTagMask &= ~NotificationTags::Important;
        } );
        UI::setTooltipIfHovered( "Show important messages about errors or warnings that could happen." );
    }

    drawMruInnerFormats_( menuWidth );

#if 0 // Hide unimplemented settings
#ifndef __EMSCRIPTEN__
    drawSeparator_( "Files and Folders" );
    // TODO
    static std::string logFolderPath = utf8string( Logger::instance().getLogFileName().parent_path() );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 1.5f * cButtonPadding * UI::scale(), cButtonPadding * UI::scale() } );
    UI::inputText( "##LogFolderPath", logFolderPath );
    ImGui::SameLine( 0, 1.5f * style.ItemInnerSpacing.x );
    if ( ImGui::Link( "Logs folder") )
        OpenDocument( asU8String( logFolderPath ) );
    ImGui::PopStyleVar();
    ImGui::SameLine( 0.0f, 0.0f );
    ImGui::SameLine( 0.0f, -30.0f * UI::scale() );
    if ( UI::button( "...", ImVec2( 24.0f * UI::scale(), 24.0f * UI::scale() ) ) )
    {
        std::filesystem::path newPath = openFolderDialog( asU8String( logFolderPath ) );
        if ( !newPath.empty() )
            logFolderPath = utf8string( newPath );
    }
#endif
#endif
}

void ViewerSettingsPlugin::drawControlTab_( float menuWidth )
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    if ( !ribbonMenu )
        return;
    drawSeparator_( "Keyboard" );

    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;
    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu )
        ribbonMenu->setShowShortcuts( true );

    drawMouseSceneControlsSettings_( menuWidth );
    drawTouchpadSettings_();
    drawSpaceMouseSettings_( menuWidth );
}

void ViewerSettingsPlugin::drawViewportTab_( float menuWidth )
{
    auto& viewport = viewer->viewport();
    const auto& viewportParameters = viewport.getParameters();
    const auto& style = ImGui::GetStyle();

    drawSeparator_( "Viewport" );

    if ( viewer->viewport_list.size() > 1 )
        ImGui::Text( "Current viewport: %d", viewport.id.value() );

    ImGui::SetNextItemWidth( 170.0f * UI::scale() );
    auto rotMode = viewportParameters.rotationMode;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    UI::combo( "Rotation Mode", ( int* )&rotMode, { "Scene Center", "Pick / Scene Center", "Pick" } );
    viewport.rotationCenterMode( rotMode );
    ImGui::PopStyleVar();

    ImGui::PushItemWidth( 80 * UI::scale() );

    bool showAxes = viewer->basisAxes->isVisible( viewport.id );
    UI::checkbox( "Show Axes", &showAxes );
    viewport.showAxes( showAxes );
    ImGui::SameLine();

    ImGui::SetCursorPosX( 155.0f * UI::scale() );
    bool showGlobalBasis = viewer->globalBasis->isVisible( viewport.id );
    UI::checkbox( "Show Global Basis", &showGlobalBasis );
    viewport.showGlobalBasis( showGlobalBasis );

    ImGui::SameLine( 310 * UI::scale() );
    bool showGlobalBasisGrid = viewer->globalBasis->isGridVisible( viewport.id );
    UI::checkboxValid( "Grid", &showGlobalBasisGrid, showGlobalBasis );
    viewer->globalBasis->setGridVisible( showGlobalBasisGrid, viewport.id );

    ImGui::PushItemWidth( 170 * UI::scale() );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    bool isAutoGlobalBasisSize = viewportParameters.globalBasisScaleMode == Viewport::Parameters::GlobalBasisScaleMode::Auto;
    if ( isAutoGlobalBasisSize )
    {
        UI::readOnlyValue<LengthUnit>( "Global Basis Scale", viewportParameters.objectScale * 0.5f );
    }
    else
    {
        auto size = viewer->globalBasis->getAxesLength( viewport.id );
        UI::drag<LengthUnit>( "Global Basis Scale", size, viewportParameters.objectScale * 0.01f, 1e-9f );
        viewer->globalBasis->setAxesProps( size, viewer->globalBasis->getAxesWidth( viewport.id ), viewport.id );
    }
    ImGui::PopStyleVar();
    ImGui::SameLine( 310 * UI::scale() );
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( cButtonPadding - cCheckboxPadding ) * UI::scale() );
    if ( UI::checkbox( "Auto", &isAutoGlobalBasisSize ) )
    {
        auto paramsCpy = viewportParameters;
        paramsCpy.globalBasisScaleMode = isAutoGlobalBasisSize ?
            Viewport::Parameters::GlobalBasisScaleMode::Auto :
            Viewport::Parameters::GlobalBasisScaleMode::Fixed;
        viewport.setParameters( paramsCpy );
    }

    bool showRotCenter = viewer->rotationSphere->isVisible( viewport.id );
    UI::checkbox( "Show Rotation Center", &showRotCenter );
    viewport.showRotationCenter( showRotCenter );

    ImGui::PopItemWidth();
    ImGui::PopStyleVar();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 2 } );

    drawProjectionModeSelector_( 170.0f * UI::scale() );
    drawBackgroundButton_( false );

    ImGui::PopStyleVar();

    auto coef = SceneSettings::get( SceneSettings::FloatType::AmbientCoefSelectedObj );
    ImGui::SetNextItemWidth( 170.0f * UI::scale() );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    if ( UI::drag<NoUnit>( "Selected Highlight Modifier", coef, 0.01f, 1.0f, 10.0f ) )
    {
        SceneSettings::set( SceneSettings::FloatType::AmbientCoefSelectedObj, coef );
    }

    UI::setTooltipIfHovered( "Ambient light brightness multiplication factor for selected objects" );
    ImGui::PopStyleVar();

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    const bool showClippingPlane = viewer->experimentalFeatures && RibbonButtonDrawer::CustomCollapsingHeader( "Clipping Plane" );
    ImGui::PopStyleVar();

    if ( showClippingPlane )
    {
        auto plane = viewportParameters.clippingPlane;
        auto showPlane = viewer->clippingPlaneObject->isVisible( viewport.id );
        plane.n = plane.n.normalized();
        auto w = ImGui::GetContentRegionAvail().x;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        ImGui::SetNextItemWidth( w );
        UI::drag<NoUnit>( "##ClippingPlaneNormal", plane.n, 1e-3f );
        ImGui::SetNextItemWidth( w / 2.0f );
        UI::drag<NoUnit>( "##ClippingPlaneD", plane.d, 1e-3f );
        ImGui::SameLine();
        ImGui::PopStyleVar();
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( cButtonPadding - cCheckboxPadding ) * UI::scale() );
        UI::checkbox( "Show##ClippingPlane", &showPlane );
        viewport.setClippingPlane( plane );
        viewport.showClippingPlane( showPlane );
    }

    drawSeparator_( "Options" );

    ImGui::SetNextItemWidth( 170.0f * UI::scale() );
    int pickRadius = int( getViewerInstance().glPickRadius );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    UI::drag<PixelSizeUnit>( "Picker Radius", pickRadius, 1, 0, 10 );
    ImGui::PopStyleVar();
    getViewerInstance().glPickRadius = uint16_t( pickRadius );
    UI::setTooltipIfHovered( "Radius of area under cursor to pick objects in scene." );

    drawSeparator_( "Defaults" );

    drawShadingModeCombo_( true, 170.0f * UI::scale() );
    drawUpDirectionSelector_();

    drawSeparator_( "Render" );

    drawRenderOptions_();
    drawShadowsOptions_( menuWidth );

    drawGlobalSettings_( 168.0f * UI::scale() );
}

void ViewerSettingsPlugin::drawMeasurementUnitsTab_()
{
    static constexpr int cMaxPrecision = 9;

    { // Common.
        drawSeparator_( "Common" );

        // --- Leading zero
        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
        bool value = UnitSettings::getShowLeadingZero();
        if ( UI::checkbox( "Leading zero", &value ) )
            UnitSettings::setShowLeadingZero( value );
        UI::setTooltipIfHovered( "If disabled, remove the lone zeroes before the decimal point." );
        ImGui::PopStyleVar();

        // --- Thousands separator

        ImGui::PushItemWidth( 170.0f * UI::scale() );
        MR_FINALLY{ ImGui::PopItemWidth(); };

        char thouSep[2] = { UnitSettings::getThousandsSeparator(), '\0' };
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( std::floor( ( ImGui::CalcItemWidth() - ImGui::CalcTextSize( thouSep ).x ) / 2 ), cButtonPadding * UI::scale() ) );
        MR_FINALLY{ ImGui::PopStyleVar(); };

        if ( UI::inputTextIntoArray( "Thousands Separator", thouSep, sizeof thouSep, ImGuiInputTextFlags_AutoSelectAll ) )
            UnitSettings::setThousandsSeparator( thouSep[0] );
        UI::setTooltipIfHovered( "A symbol used to separate groups of thousands in large numbers to make them easier to read." );

        // If the separator is empty or a space, display a string explaining that on top of the textbox.
        if ( !ImGui::IsItemActive() )
        {
            const char* label = nullptr;
            if ( thouSep[0] == 0 )
                label = "None";
            else if ( thouSep[0] == ' ' )
                label = "Space";

            if ( label )
            {
                ImVec2 textSize = ImGui::CalcTextSize( label );
                ImGui::GetWindowDrawList()->AddText( ImGui::GetItemRectMin() + ( ImVec2( ImGui::CalcItemWidth(), ImGui::GetItemRectSize().y ) - textSize ) / 2, ImGui::GetColorU32( ImGuiCol_TextDisabled ), label );
            }
        }
    }


    { // Length.
        ImGui::PushItemWidth( 170.0f * UI::scale() );
        drawSeparator_( "Linear" );

        ImGui::PushID( "length" );
        MR_FINALLY{ ImGui::PopID(); };

        // --- Units

        auto makeLengthUnitsVec = []( const char * lastOption )
        {
            std::vector<std::string> ret;
            ret.reserve( std::size_t( LengthUnit::_count ) + 1 );
            for ( std::size_t i = 0; i < std::size_t( LengthUnit::_count ); i++ )
                ret.emplace_back( getUnitInfo( LengthUnit( i ) ).prettyName );
            ret.emplace_back( lastOption );
            return ret;
        };

        int targetOption = int( UnitSettings::getUiLengthUnit().value_or( LengthUnit::_count ) );
        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        static const std::vector<std::string> uiLengthUnitNames = makeLengthUnitsVec( "No Units" );
        if ( UI::combo( "UI Units##length", &targetOption, uiLengthUnitNames ) )
        {
            if ( targetOption == int( LengthUnit::_count ) )
                UnitSettings::setUiLengthUnit( {}, true );
            else
                UnitSettings::setUiLengthUnit( LengthUnit( targetOption ), true );
        }
        UI::setTooltipIfHovered( "It selects length units to be show in the user interface. If model units are different, then stored values will be automatically converted when shown in UI." );

        int sourceOption = int( UnitSettings::getModelLengthUnit().value_or( LengthUnit::_count ) );
        static const std::vector<std::string> modelLengthUnitNames = makeLengthUnitsVec( "Same as UI Units" );
        if ( UI::combo( "Model Units##length", &sourceOption, modelLengthUnitNames ) )
        {
            if ( sourceOption == int( LengthUnit::_count ) )
                UnitSettings::setModelLengthUnit( {} );
            else
                UnitSettings::setModelLengthUnit( LengthUnit( sourceOption ) );
        }
        UI::setTooltipIfHovered( "It selects length units of model's actual values (e.g. coordinates of points stored in memory). And it affects on importing and exporting of data." );

        // --- Precision
        int precision = UnitSettings::getUiLengthPrecision();
        if ( UI::drag<NoUnit>( "Precision##length", precision, 1, 0, cMaxPrecision ) )
            UnitSettings::setUiLengthPrecision( precision );
        UI::setTooltipIfHovered( "The number of digits to be shown after decimal point for length measurements." );

        ImGui::PopStyleVar();
        ImGui::PopItemWidth();
    }

    { // Angle.
        ImGui::PushItemWidth( 170.0f * UI::scale() );
        drawSeparator_( "Angular" );

        static const std::vector<std::string> flavorOptions = []{
            std::vector<std::string> ret;
            ret.reserve( std::size_t( DegreesMode::_count ) );
            for ( std::size_t i = 0; i < std::size_t( DegreesMode::_count ); i++ )
                ret.emplace_back( toString( DegreesMode( i ) ) );
            return ret;
        }();

        int flavorOption = int( UnitSettings::getDegreesMode() );

        // Degree mode.
        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        if ( UI::combo( "Units##angle", &flavorOption, flavorOptions ) )
            UnitSettings::setDegreesMode( DegreesMode( flavorOption ), true );
        UI::setTooltipIfHovered( "It selects angular units to be show in the user interface." );

        // Degree-mode-specific options.

        if ( getDefaultUnitParams<AngleUnit>().degreesMode == DegreesMode::degrees )
        {
            // --- Precision

            int precision = UnitSettings::getUiAnglePrecision();
            if ( UI::drag<NoUnit>( "Precision##angle", precision, 1, 0, cMaxPrecision ) )
                UnitSettings::setUiAnglePrecision( precision );
            UI::setTooltipIfHovered( "The number of digits to be shown after decimal point for angular measurements." );
        }

        ImGui::PopStyleVar();
        ImGui::PopItemWidth();
    }

    { // Ratio.
        ImGui::PushItemWidth( 170.0f * UI::scale() );
        drawSeparator_( "Scale and Ratios" );

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, cButtonPadding * UI::scale() } );

        // --- Precision

        int precision = UnitSettings::getUiRatioPrecision();
        if ( UI::drag<NoUnit>( "Precision##ratio", precision, 1, 0, cMaxPrecision ) )
            UnitSettings::setUiRatioPrecision( precision );
        UI::setTooltipIfHovered( "The number of digits to be shown after decimal point for dimensionless measurements." );

        ImGui::PopStyleVar();
        ImGui::PopItemWidth();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if ( UI::button( "Reset Unit Settings" ) )
        UnitSettings::resetToDefaults();
    UI::setTooltipIfHovered( "Set all settings here to their default values." );
}

void ViewerSettingsPlugin::drawFeaturesTab_()
{
    const auto& style = ImGui::GetStyle();

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    MR_FINALLY{ ImGui::PopStyleVar(); };
    ImGui::PushItemWidth( 200.0f * UI::scale() );
    MR_FINALLY{ ImGui::PopItemWidth(); };

    float value = 0;

    drawSeparator_( "Visuals" );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureMeshAlpha );
    if ( UI::slider<NoUnit>( "Surface opacity", value, 0.f, 1.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureMeshAlpha, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeaturePointSize );
    if ( UI::slider<PixelSizeUnit>( "Point size", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeaturePointSize, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureSubPointSize );
    if ( UI::slider<PixelSizeUnit>( "Point size (subfeatures)", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureSubPointSize, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureLineWidth );
    if ( UI::slider<PixelSizeUnit>( "Line width", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureLineWidth, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureSubLineWidth );
    if ( UI::slider<PixelSizeUnit>( "Line width (subfeatures)", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureSubLineWidth, value );
}

void ViewerSettingsPlugin::drawRenderOptions_()
{
    auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );

    if ( viewer->isAlphaSortAvailable() )
    {
        bool alphaSortBackUp = viewer->isAlphaSortEnabled();
        bool alphaBoxVal = alphaSortBackUp;
        UI::checkbox( "Alpha Sort", &alphaBoxVal );
        if ( alphaBoxVal != alphaSortBackUp )
            viewer->enableAlphaSort( alphaBoxVal );
    }

    ImGui::PushItemWidth( 100.0f * UI::scale() );
    if ( viewer->isAlphaSortEnabled() )
    {
        UI::readOnlyValue<NoUnit>( "Depth Peeling Passes", viewer->getDepthPeelNumPasses() );
    }
    else
    {
        int dpNumPasses = viewer->getDepthPeelNumPasses();
        UI::input<NoUnit>( "Depth Peeling Passes", dpNumPasses, 0, 64 );
        viewer->setDepthPeelNumPasses( dpNumPasses );
    }
    ImGui::PopItemWidth();

    if ( viewer->isGLInitialized() )
    {
        if ( maxSamples_ > 1 )
        {
            auto backUpSamples = viewer->getRequestedMSAA();
            auto newSamples = backUpSamples;
            ImGui::Text( "Multisample anti-aliasing (MSAA):" );
            UI::setTooltipIfHovered( "The number of samples per pixel: more samples - better render quality but worse performance." );
            int counter = 0;
            for ( int i = 0; i <= maxSamples_; i <<= 1 )
            {
#ifdef __EMSCRIPTEN__
                if ( !viewer->isSceneTextureEnabled() && i == 2 )
                    continue; // only OFF and x4 are available for main framebuffer in web
#endif
                if ( i == 0 )
                {
                    UI::radioButton( "Off", &newSamples, i );
                    ++i;
                }
                else
                {
                    std::string label = 'x' + std::to_string( i );
                    UI::radioButton( label.c_str(), &newSamples, i );
                }
                if ( i << 1 <= maxSamples_ )
                    ImGui::SameLine( ( ( ++counter ) * 70.f + style.WindowPadding.x ) * UI::scale() );
            }
            if ( newSamples != backUpSamples )
                viewer->requestChangeMSAA( newSamples );
            int initMSAA = storedSamples_;
            int actualMSAA = viewer->getMSAA();
            int requestedMSAA = viewer->getRequestedMSAA();
            if ( actualMSAA != requestedMSAA )
            {
                if ( gpuOverridesMSAA_ )
                    UI::transparentTextWrapped( "GPU multisampling settings override application value." );
                if ( requestedMSAA != initMSAA && !viewer->isSceneTextureEnabled() )
                    UI::transparentTextWrapped( "Application requires restart to apply this change" );
            }
        }
    }

    ImGui::PopStyleVar();
}

void ViewerSettingsPlugin::drawShadowsOptions_( float )
{
    const auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    const bool showShadows = shadowGl_ && RibbonButtonDrawer::CustomCollapsingHeader( "Shadows" );
    ImGui::PopStyleVar();

    if ( showShadows )
    {
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + cDefaultItemSpacing * UI::scale() * 0.5f );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
        bool isEnableShadows = shadowGl_->isEnabled();
        UI::checkbox( "Enabled", &isEnableShadows );
        if ( isEnableShadows != shadowGl_->isEnabled() )
        {
            CommandLoop::appendCommand( [shadowGl = shadowGl_.get(), isEnableShadows] ()
            {
                shadowGl->enable( isEnableShadows );
            } );
        }
        ImGui::SameLine( 116.0f * UI::scale() );
        auto color = shadowGl_->getShadowColor();
        UI::colorEdit4( "Shadow Color", color,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
        shadowGl_->setShadowColor( color );
        ImGui::PopStyleVar();

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * 1.5f } );
        ImGui::PushItemWidth( 208.0f * UI::scale() );
        auto shift = shadowGl_->getShadowShift();
        auto radius = shadowGl_->getBlurRadius();
        auto quality = shadowGl_->getQuality();
        UI::drag<PixelSizeUnit>( "Shift", shift, 0.4f, -200.0f, 200.0f );
        ImGui::SetItemTooltip( "X = shift to the left, Y = shift upwards" );
        UI::drag<PixelSizeUnit>( "Blur Radius", radius, 0.2f, 0.f, 200.f );
        UI::drag<NoUnit>( "Quality", quality, 0.001f, 0.0625f, 1.0f );
        ImGui::PopItemWidth();
        ImGui::PopStyleVar();
        UI::setTooltipIfHovered( "Blur texture downscaling coefficient" );
        shadowGl_->setShadowShift( shift );
        shadowGl_->setBlurRadius( radius );
        shadowGl_->setQuality( quality );
    }
}

void ViewerSettingsPlugin::drawThemeSelector_()
{
    const auto& style = ImGui::GetStyle();

    ImGui::SetNextItemWidth( 200.0f * UI::scale() );
    int selectedUserIdxBackup = selectedUserPreset_;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    UI::combo( "Color Theme", &selectedUserPreset_, userThemesPresets_ );
    ImGui::PopStyleVar();
    if ( selectedUserPreset_ != selectedUserIdxBackup )
    {
        if ( selectedUserPreset_ == 0 )
            ColorTheme::setupDefaultDark();
        else if ( selectedUserPreset_ == 1 )
            ColorTheme::setupDefaultLight();
        else
            ColorTheme::setupUserTheme( userThemesPresets_[selectedUserPreset_] );
        if ( !ColorTheme::isInitialized() )
        {
            if ( selectedUserIdxBackup == 0 )
                ColorTheme::setupDefaultDark();
            else if ( selectedUserIdxBackup == 1 )
                ColorTheme::setupDefaultLight();
            else
                ColorTheme::setupUserTheme( userThemesPresets_[selectedUserIdxBackup] );
            selectedUserPreset_ = selectedUserIdxBackup;

            showError( "This theme is not valid." );
        }
        backgroundColor_ = Vector4f( ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::Background ) );
    }
    auto item = RibbonSchemaHolder::schema().items.find( "Add custom theme" );
    if ( item != RibbonSchemaHolder::schema().items.end() )
    {
        ImGui::SameLine( 300.0f * UI::scale() );
        if ( UI::button( "Add",
            item->second.item->isAvailable( getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) ).empty(),
            Vector2f( 50.0f * UI::scale(), 0 ) ) )
        {
            item->second.item->action();
        }
        UI::setTooltipIfHovered( item->second.tooltip );
    }
}

void ViewerSettingsPlugin::drawResetDialog_( bool activated )
{
    if ( activated )
        ImGui::OpenPopup( "Settings reset" );
    ModalDialog dialog( "Settings reset", {
        .text = "Reset all application settings?",
    } );
    if ( dialog.beginPopup() )
    {
        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );

        const float p = style.ItemSpacing.x;
        const Vector2f btnSize{ ( ImGui::GetContentRegionAvail().x - p ) / 2.f, 0 };

        if ( UI::buttonCommonSize( "Reset", btnSize, ImGuiKey_Enter ) )
        {
            resetSettings_();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine( 0, p );
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        ImGui::PopStyleVar(); // ImGuiStyleVar_FramePadding
        dialog.endPopup();
    }
}

void ViewerSettingsPlugin::drawShadingModeCombo_( bool inGroup, float toolWidth )
{
    const auto& style = ImGui::GetStyle();

    static std::vector<std::string> shadingModes = { "Auto Detect", "Smooth", "Flat" };
    SceneSettings::ShadingMode shadingMode = SceneSettings::getDefaultShadingMode();
    ImGui::SetNextItemWidth( toolWidth );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    UI::combo( inGroup ? "Shading Mode" : "Default Shading Mode", ( int* )&shadingMode, shadingModes);
    ImGui::PopStyleVar();
    UI::setTooltipIfHovered( "Shading mode for mesh objects imported from files\n"
        "Detection depends on source format and mesh shape\n"
        "This setting also affects some tools" );
    if ( shadingMode != SceneSettings::getDefaultShadingMode() )
        SceneSettings::setDefaultShadingMode( shadingMode );
}

void ViewerSettingsPlugin::drawProjectionModeSelector_( float toolWidth )
{
    const auto& style = ImGui::GetStyle();

    ImGui::SetNextItemWidth( toolWidth );
    static std::vector<std::string> projectionModes = { "Orthographic", "Perspective" };
    int projectionMode = viewer->viewport().getParameters().orthographic ? 0 : 1;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    if ( UI::combo( "Projection Mode", &projectionMode, projectionModes) )
        viewer->viewport().setOrthographic( projectionMode == 0 );
    ImGui::PopStyleVar();
}

void ViewerSettingsPlugin::drawUpDirectionSelector_()
{
#if 0 // Hide unimplemented settings
    // TODO
    if ( !viewer->experimentalFeatures )
        return;
    ImGui::Text( "Up Direction" );
    static int axis = 2; // Z
    ImGui::SameLine();
    UI::radioButton( "Y", &axis, 1 );
    ImGui::SameLine();
    UI::radioButton( "Z", &axis, 2 );
#endif
}

void ViewerSettingsPlugin::drawBackgroundButton_( bool allViewports )
{
    bool needUpdateBackup = backgroundColor_.w == -1.0f;
    if ( needUpdateBackup )
        backgroundColor_ = Vector4f( viewer->viewport().getParameters().backgroundColor );

    auto backgroundColor = backgroundColor_;

    if ( UI::colorEdit4( "Background Color", backgroundColor,
        ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
        backgroundColor_ = backgroundColor;
    else if ( ImGui::IsWindowFocused() || !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) )
        backgroundColor_.w = -1.0f;
    if ( allViewports )
        for ( ViewportId vid : viewer->getPresentViewports() )
            viewer->viewport( vid ).setBackgroundColor( Color( backgroundColor ) );
    else
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );
}

void ViewerSettingsPlugin::drawMouseSceneControlsSettings_( float menuWidth )
{
    const auto& style = ImGui::GetStyle();

    drawSeparator_( "Mouse" );

    ImGui::SetNextItemWidth( 100 * UI::scale() );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    UI::drag<NoUnit>( "Zoom Gain", viewer->scrollForce, 0.01f, 0.2f, 3.0f );
    ImGui::PopStyleVar();
    UI::setTooltipIfHovered( "Sensitivity for mouse wheel rotation affecting the speed of zooming." );

    UI::separator( UI::SeparatorParams{ .extraScale = cSeparatorIndentMultiplier } );

    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        if ( mode == MouseMode::None )
            continue;
        std::string modeName = getMouseModeString( mode );
        std::string ctrlStr = "None";
        auto ctrl = viewer->mouseController().findControlByMode( mode );
        if ( ctrl )
            ctrlStr = MouseController::getControlString( *ctrl );

        const float posY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( posY + cRibbonButtonWindowPaddingY * UI::scale() / 2.f );
        ImGui::Text( "%s", modeName.c_str() );

        ImGui::SetCursorPosX( 110.0f * UI::scale() );
        ImGui::SetCursorPosY( posY - cRibbonButtonWindowPaddingY * UI::scale() / 2.f );

        auto plusPos = ctrlStr.rfind( '+' );
        std::string uniqueKeyStr = "##key" + std::to_string( i );
        if ( plusPos == std::string::npos )
        {
            // Draw button name in a frame
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, ( cRibbonButtonWindowPaddingY + 1 ) * UI::scale() } );
            UI::inputTextCenteredReadOnly( uniqueKeyStr.c_str(), ctrlStr, 54 * UI::scale() );
            ImGui::PopStyleVar();
            ImGui::SameLine();
        }
        else
        {
            // Draw modifier and key in separate frames
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x * 0.25f, style.ItemSpacing.y } );

            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, ( cRibbonButtonWindowPaddingY + 1 ) * UI::scale() } );
            auto modKeyUniqueStr = "##modifierKey" + std::to_string( i );
            UI::inputTextCenteredReadOnly( modKeyUniqueStr.c_str(), ctrlStr.substr( 0, plusPos ),
                // Expand the area in case of multiple modifiers (assume that is rarely used)
                std::max( 54.0f, 7 * float( plusPos ) ) * UI::scale() );
            ImGui::PopStyleVar();

            ImGui::SameLine();
            ImGui::SetCursorPosY( posY - cRibbonButtonWindowPaddingY * UI::scale() / 2.f );
            ImGui::Text( "+" );
            ImGui::SameLine();
            ImGui::SetCursorPosY( posY - cRibbonButtonWindowPaddingY * UI::scale() / 2.f );

            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, ( cRibbonButtonWindowPaddingY + 1 ) * UI::scale() } );
            UI::inputTextCenteredReadOnly( uniqueKeyStr.c_str(), ctrlStr.substr( plusPos + 1 ), 54 * UI::scale() );
            ImGui::PopStyleVar();

            ImGui::PopStyleVar();
        }

        ImGui::SetCursorPosX( menuWidth - 120.0f * UI::scale() );
		ImGui::SetCursorPosY( posY - cRibbonButtonWindowPaddingY * UI::scale() / 2.f );
        UI::buttonCommonSize( fmt::format( "Set other##{}", i ).c_str(), Vector2f( 80 * UI::scale(), 0 ) );
        if ( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ctrlStr = ShortcutManager::getModifierString( getGlfwModPrimaryCtrl() );
            ImGui::Text( "Click here with preferred mouse button \nwith/without modifier (%s/%s/Shift)", getAltModName(), ctrlStr.c_str() );
            ImGui::EndTooltip();

            if ( ImGui::GetIO().MouseClicked[0] || ImGui::GetIO().MouseClicked[1] || ImGui::GetIO().MouseClicked[2] )
            {
                MouseButton clikedBtn = MouseButton::Left;
                if ( ImGui::GetIO().MouseClicked[1] )
                    clikedBtn = MouseButton::Right;
                else if ( ImGui::GetIO().MouseClicked[2] )
                    clikedBtn = MouseButton::Middle;

                int modifier = 0;
                if ( ImGui::IsKeyDown( UI::getImGuiModPrimaryCtrl() ) )
                    modifier |= getGlfwModPrimaryCtrl();
                if ( ImGui::GetIO().KeyAlt )
                    modifier |= GLFW_MOD_ALT;
                if ( ImGui::GetIO().KeyShift )
                    modifier |= GLFW_MOD_SHIFT;

                viewer->mouseController().setMouseControl( { clikedBtn,modifier }, mode );
            }
        }
    }

    std::string keysListWithAlt;
    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        auto ctrl = viewer->mouseController().findControlByMode( mode );
        if ( !ctrl || ( ctrl->mod & GLFW_MOD_ALT ) != 0 )
            continue;
        MouseController::MouseControlKey ctrlAlt = *ctrl;
        ctrlAlt.mod |= GLFW_MOD_ALT;
        if ( !keysListWithAlt.empty() )
            keysListWithAlt += ", ";
        keysListWithAlt += MouseController::getControlString( ctrlAlt );
    }
    UI::transparentTextWrapped( "Camera controls can also be used with %s", getAltModName() );
    if ( !keysListWithAlt.empty() )
        UI::setTooltipIfHovered( keysListWithAlt );
}

void ViewerSettingsPlugin::drawSpaceMouseSettings_( float menuWidth )
{
    drawSeparator_( "Spacemouse" );

    bool anyChanged = false;
    auto drawSlider = [&anyChanged, menuWidth] ( const char* label, float& value )
    {
        int valueAbs = int( std::fabs( value ) );
        bool inverse = value < 0.f;
        ImGui::SetNextItemWidth( menuWidth * 0.6f );
        bool changed = UI::slider<NoUnit>( label, valueAbs, 1, 100 );
        ImGui::SameLine( menuWidth * 0.78f );
        const float cursorPosY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( cursorPosY + ( cInputPadding - cCheckboxPadding ) * UI::scale() );
        changed = UI::checkbox( ( std::string( "Inverse##" ) + label ).c_str(), &inverse ) || changed;
        if ( changed )
            value = valueAbs * ( inverse ? -1.f : 1.f );
        anyChanged = anyChanged || changed;
    };

    const auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
    ImGui::Text( "%s", "Translation Sensitivity" );
    ImGui::PopStyleVar();

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );

    drawSlider( "X##translate", spaceMouseParams_.translateScale[0] );
    drawSlider( "Y##translate", spaceMouseParams_.translateScale[2] );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 2.0f } );
    drawSlider( "Zoom##translate", spaceMouseParams_.translateScale[1] );
    ImGui::PopStyleVar();

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
    ImGui::Text( "%s", "Rotation Sensitivity" );
    ImGui::PopStyleVar();

    drawSlider( "Ox##rotate", spaceMouseParams_.rotateScale[0] );
    drawSlider( "Oy##rotate", spaceMouseParams_.rotateScale[1] );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 2.0f } );
    drawSlider( "Oz##rotate", spaceMouseParams_.rotateScale[2] );
    ImGui::PopStyleVar( 2 );

#if defined(_WIN32) || defined(__APPLE__)
    if ( UI::checkbox( "Zoom by mouse wheel", &activeMouseScrollZoom_ ) )
    {
        if ( auto spaceMouseHandler = getViewerInstance().getSpaceMouseHandler() )
        {
            auto hidapiHandler = std::dynamic_pointer_cast< SpaceMouseHandlerHidapi >( spaceMouseHandler );
            if ( hidapiHandler )
            {
                hidapiHandler->activateMouseScrollZoom( activeMouseScrollZoom_ );
            }
        }
    }
    UI::setTooltipIfHovered( "This mode is NOT recommended if you have 3Dconnexion driver installed, which sends mouse wheel fake events resulting in double reaction on SpaceMouse movement and camera tremble." );
#endif
    if ( anyChanged )
        getViewerInstance().setSpaceMouseParameters( spaceMouseParams_ );
}

void ViewerSettingsPlugin::drawTouchpadSettings_()
{
    const auto& style = ImGui::GetStyle();

    drawSeparator_( "Touchpad" );

    const std::vector<std::string> swipeModeList = { "Swipe Rotates Camera", "Swipe Moves Camera" };
    assert( swipeModeList.size() == (size_t)TouchpadParameters::SwipeMode::Count );

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, style.ItemSpacing.y * 1.5f } );
    bool updateSettings = false;
    if ( UI::checkbox( "Ignore Kinetic Movements", &touchpadParameters_.ignoreKineticMoves ) )
        updateSettings = true;
    if ( UI::checkbox( "Allow System to Interrupt Gestures", &touchpadParameters_.cancellable ) )
        updateSettings = true;
    ImGui::PopStyleVar();

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
    if ( UI::combo( "Swipe Mode", (int*)&touchpadParameters_.swipeMode, swipeModeList ) )
        updateSettings = true;
    ImGui::PopStyleVar();
    if ( updateSettings )
        viewer->setTouchpadParameters( touchpadParameters_ );
}

void ViewerSettingsPlugin::drawMruInnerFormats_( float menuWidth )
{
    drawSeparator_( "MRU Inner Formats" );

    const std::vector<std::string> meshFormatNames = { "CTM", "PLY", "MRMESH" };
    const std::vector<std::string> pointsFormatNames = { meshFormatNames[0], meshFormatNames[1] };
    const std::vector<std::string> voxelsFormatNames = { "VDB", "RAW" };

    const std::vector<std::string> meshFormatTooltips = { "Slowest, high memory consumption, but best compression (typically) format",
                                                    "Fast and still relatively small format",
                                                    "Largest by size, but fastest to load / save and without any losses" };
    const std::vector<std::string> pointsFormatTooltips = { meshFormatTooltips[0], meshFormatTooltips[1] };
    const std::vector<std::string> voxelsFormatTooltips = { "Fast and efficient format for sparse data",
                                                            "Simplest but high disk space consumption format" };

    std::string format = defaultSerializeMeshFormat();
    if ( format == ".ctm" )
        mruFormatParameters_.meshFormat = MruFormatParameters::MeshFormat::Ctm;
    else if ( format == ".mrmesh" )
        mruFormatParameters_.meshFormat = MruFormatParameters::MeshFormat::Mrmesh;
    else // format == ".ply"
        mruFormatParameters_.meshFormat = MruFormatParameters::MeshFormat::Ply;

    format = defaultSerializePointsFormat();
    if ( format == ".ctm" )
        mruFormatParameters_.pointsFormat = MruFormatParameters::PointsFormat::Ctm;
    else // format == ".ply"
        mruFormatParameters_.pointsFormat = MruFormatParameters::PointsFormat::Ply;

    #ifndef MRVIEWER_NO_VOXELS
    format = defaultSerializeVoxelsFormat();
    if ( format == ".raw" )
        mruFormatParameters_.voxelsFormat = MruFormatParameters::VoxelsFormat::Raw;
    else // format == ".vdb"
        mruFormatParameters_.voxelsFormat = MruFormatParameters::VoxelsFormat::Vdb;
    #endif

    ImGui::PushItemWidth( menuWidth * 0.5f );
    if ( UI::combo( "Mesh Format", ( int* )&mruFormatParameters_.meshFormat, meshFormatNames, true, meshFormatTooltips ) )
    {
        switch ( mruFormatParameters_.meshFormat )
        {
        case MruFormatParameters::MeshFormat::Ctm:
            format = ".ctm";
            break;
        case MruFormatParameters::MeshFormat::Mrmesh:
            format = ".mrmesh";
            break;
        case MruFormatParameters::MeshFormat::Ply:
        default:
            format = ".ply";
            break;
        }
        setDefaultSerializeMeshFormat( format );
    }

    if ( UI::combo( "Points Format", ( int* )&mruFormatParameters_.pointsFormat, pointsFormatNames, true, pointsFormatTooltips ) )
    {
        switch ( mruFormatParameters_.pointsFormat )
        {
        case MruFormatParameters::PointsFormat::Ctm:
            format = ".ctm";
            break;
        case MruFormatParameters::PointsFormat::Ply:
        default:
            format = ".ply";
            break;
        }
        setDefaultSerializePointsFormat( format );
    } 
    #ifndef MRVIEWER_NO_VOXELS
    if ( UI::combo( "Voxels Format", ( int* )&mruFormatParameters_.voxelsFormat, voxelsFormatNames, true, voxelsFormatTooltips ) )
    {
        switch ( mruFormatParameters_.voxelsFormat )
        {
        case MruFormatParameters::VoxelsFormat::Raw:
            format = ".raw";
            break;
        case MruFormatParameters::VoxelsFormat::Vdb:
        default:
            format = ".vdb";
            break;
        }
        setDefaultSerializeVoxelsFormat( format );
    }
    #endif
    ImGui::PopItemWidth();
}

void ViewerSettingsPlugin::drawCustomSettings_( const std::string& separatorName, bool needSeparator )
{
    if ( comboSettings_[size_t( activeTab_ )].empty() )
        return;
    int numRequired = 0;
    for ( auto& settings : comboSettings_[size_t( activeTab_ )] )
    {
        if ( settings->separatorName() == separatorName )
            ++numRequired;
    }
    if ( numRequired == 0 )
        return;
    if ( needSeparator )
        UI::separator( UI::SeparatorParams{ .label = separatorName, .extraScale = cSeparatorIndentMultiplier } );
    for ( auto& settings : comboSettings_[size_t( activeTab_ )] )
    {
        if ( settings->separatorName() == separatorName )
            settings->draw();
    }
}


void ViewerSettingsPlugin::drawSeparator_( const std::string& separatorName )
{
    UI::separator( UI::SeparatorParams{ .label = separatorName, .extraScale = cSeparatorIndentMultiplier } );
    drawCustomSettings_( separatorName, false );
}

void ViewerSettingsPlugin::updateDialog_()
{
    orderedTab_ = TabType::Count;
    updateThemes();

    tempUserScaling_ = viewer->getMenuPlugin()->getUserScaling();
    spaceMouseParams_ = viewer->getSpaceMouseParameters();
    touchpadParameters_ = viewer->getTouchpadParameters();
#if defined(_WIN32) || defined(__APPLE__)
    if ( auto spaceMouseHandler = viewer->getSpaceMouseHandler() )
    {
        auto hidapiHandler = std::dynamic_pointer_cast< MR::SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapiHandler )
            activeMouseScrollZoom_ = hidapiHandler->isMouseScrollZoomActive();
    }
#endif
}

void ViewerSettingsPlugin::resetSettings_()
{
    viewer->getViewerSettingsManager()->resetSettings( *viewer );

    for ( size_t tabType = size_t( 0 ); tabType < size_t( TabType::Count ); tabType++ )
        for ( auto& settings : comboSettings_[ tabType ] )
            settings->reset();

    CommandLoop::appendCommand( [shadowGl = shadowGl_.get()] ()
    {
        shadowGl->enable( false );
    } );

    if ( auto& settingsManager = viewer->getViewerSettingsManager() )
        settingsManager->saveString( "multisampleAntiAliasing", "invalid" );// invalidate record, so next time - default value will be used

#if defined(_WIN32) || defined(__APPLE__)
    if ( auto spaceMouseHandler = viewer->getSpaceMouseHandler() )
    {
        auto hidapiHandler = std::dynamic_pointer_cast< MR::SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapiHandler )
            hidapiHandler->activateMouseScrollZoom( false );
    }
#endif

    updateDialog_();
}

MR_REGISTER_RIBBON_ITEM( ViewerSettingsPlugin )

}
