#include "MRViewerSettingsPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRColorTheme.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewerSettingsManager.h"
#include "MRViewer/MRGLMacro.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRSpaceMouseHandlerHidapi.h"
#include "MRMesh/MRLog.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRUIStyle.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRDirectory.h"
#include <MRMesh/MRSceneRoot.h>
#include <MRViewer/MRFileDialog.h>
#include "MRMesh/MRObjectMesh.h"

namespace
{
const char* getViewerSettingTabName( MR::ViewerSettingsPlugin::TabType tab )
{
    constexpr std::array<const char*, size_t( MR::ViewerSettingsPlugin::TabType::Count )> tabNames{
        "Quick",
        "Application",
        "Control",
        "3D View",
        "Features"      // Reserved for custom content
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
    CommandLoop::appendCommand( [maxSamples = &maxSamples_, curSamples = &curSamples_, storedSamples = &storedSamples_] ()
    {
        if ( getViewerInstance().isGLInitialized() && loadGL() )
        {
            GL_EXEC( glGetIntegerv( GL_MAX_SAMPLES, maxSamples ) );
            GL_EXEC( glGetIntegerv( GL_SAMPLES, curSamples ) );
            *maxSamples = std::max( std::min( *maxSamples, 16 ), *curSamples ); // there are some known issues with 32 MSAA
            *storedSamples = *curSamples;
        }
    } );
#ifndef __EMSCRIPTEN__
    CommandLoop::appendCommand( [&] ()
    {
        auto& viewer = getViewerInstance();
        int samples = 0;
        if ( auto& settingsManager = viewer.getViewportSettingsManager() )
            samples = settingsManager->loadInt( "multisampleAntiAliasing", 8 );
        if ( viewer.isGLInitialized() && loadGL() )
        {
            int realSamples;
            GL_EXEC( glGetIntegerv( GL_SAMPLES, &realSamples ) );
            gpuOverridesMSAA_ = ( realSamples != samples );
        }
    }, CommandLoop::StartPosition::AfterWindowAppear );
#endif
}

const std::string& ViewerSettingsPlugin::uiName() const
{
    static std::string name = "Settings";
    return name;
}

void ViewerSettingsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 500.0f * menuScaling;

    ImVec2 position{ ( viewer->framebufferSize.x - menuWidth ) / 2, viewer->framebufferSize.y / 6.0f };
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .position = &position, .menuScaling = menuScaling } ) )
        return;

    if ( UI::beginTabBar( "##MainTabs" ) )
    {
        for ( int i = 0; i<int( TabType::Count ); ++i )
        {
            if ( i == int( TabType::Features ) && !RibbonSchemaHolder::schema().experimentalFeatures )
                continue;
            auto tab = TabType( i );
            if ( UI::beginTabItem( getViewerSettingTabName( tab ) ) )
            {
                activeTab_ = tab;
                drawTab_( tab, menuWidth, menuScaling );
                drawCustomSettings_( tab, menuScaling );
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

bool ViewerSettingsPlugin::onEnable_()
{
    backgroundColor_.w = -1.0f;

    ribbonMenu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>().get();
    updateThemes();

    auto& viewerRef = getViewerInstance();
    spaceMouseParams_ = viewerRef.getSpaceMouseParameters();
#if defined(_WIN32) || defined(__APPLE__)
    if ( auto spaceMouseHandler = viewerRef.getSpaceMouseHandler() )
    {
        auto hidapiHandler = std::dynamic_pointer_cast< MR::SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapiHandler )
            activeMouseScrollZoom_ = hidapiHandler->isMouseScrollZoomActive();
    }
#endif

    return true;
}

bool ViewerSettingsPlugin::onDisable_()
{
    userThemesPresets_.clear();
    ribbonMenu_ = nullptr;
    return true;
}

void ViewerSettingsPlugin::drawTab_( TabType tab, float menuWidth, float menuScaling )
{
    switch ( tab )
    {
    case MR::ViewerSettingsPlugin::TabType::Quick:
        drawQuickTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Application:
        drawApplicationTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Control:
        drawControlTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Viewport:
        drawViewportTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Features:
        // Custom controls only
        break;
    case MR::ViewerSettingsPlugin::TabType::Count:
    default:
        break;
    }
}

void ViewerSettingsPlugin::drawQuickTab_( float menuWidth, float menuScaling )
{
    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;

    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->openToolbarCustomize();
    ImGui::SameLine();
    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->setShowShortcuts( true );

    drawThemeSelector_( menuWidth, menuScaling );
    drawProjectionModeSelector_( false, menuScaling );
    drawShadingModeCombo_( false, menuScaling );
    drawBackgroundButton_( true );
    drawMouseSceneControlsSettings_( false, menuWidth, menuScaling );
}

void ViewerSettingsPlugin::drawApplicationTab_( float menuWidth, float menuScaling )
{
    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;

    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->openToolbarCustomize();

    drawThemeSelector_( menuWidth, menuScaling );

    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Behavior", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::SetNextItemWidth( menuWidth * 0.5f );
        if ( ribbonMenu_ )
        {
            UI::checkbox( "Make Visible on Select",
                                                  std::bind( &RibbonMenu::getShowNewSelectedObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setShowNewSelectedObjects, ribbonMenu_, std::placeholders::_1 ) );
            UI::checkbox( "Deselect on Hide",
                                                  std::bind( &RibbonMenu::getDeselectNewHiddenObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setDeselectNewHiddenObjects, ribbonMenu_, std::placeholders::_1 ) );
            UI::checkbox( "Close Context Menu on Change",
                                                  std::bind( &RibbonMenu::getCloseContextOnChange, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setCloseContextOnChange, ribbonMenu_, std::placeholders::_1 ) );
            UI::setTooltipIfHovered( "Close scene context menu on any change", menuScaling );

            UI::checkbox( "Close Tool on Activating Another One",
                                                  std::bind( &RibbonMenu::getAutoCloseBlockingPlugins, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setAutoCloseBlockingPlugins, ribbonMenu_, std::placeholders::_1 ) );
            UI::setTooltipIfHovered( "Automatically close blocking tool when another blocking tool is activated", menuScaling );

            UI::checkbox( "Show Experimental Features", &RibbonSchemaHolder::schema().experimentalFeatures );
            UI::setTooltipIfHovered( "Show experimental ribbon tabs", menuScaling );
        }
    }
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Interface", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        // TODO
        static int decimalPlaces = 2;
        if ( !viewer->isDeveloperFeaturesEnabled() )
            goto skip;
        ImGui::SetNextItemWidth( 100.0f * menuScaling );
        ImGui::DragInputInt( "Decimal places", &decimalPlaces, 1, 0, 10 );
        UI::setTooltipIfHovered( "Show this number of digits after decimal dot", menuScaling );
        skip:

        bool savedDialogsBackUp = viewer->getMenuPlugin()->isSavedDialogPositionsEnabled();
        bool savedDialogsVal = savedDialogsBackUp;
        UI::checkbox( "Save Tool Window Positions", &savedDialogsVal );
        UI::setTooltipIfHovered( "If checked then enables using of saved positions of tool windows in the config file", menuScaling );
        if ( savedDialogsVal != savedDialogsBackUp )
            viewer->getMenuPlugin()->enableSavedDialogPositions( savedDialogsVal );
    }
    if ( !viewer->isDeveloperFeaturesEnabled() )
        return; // TODO
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Notifications" ) )
    {
        static bool newVersion, importWarnings; // TODO
        UI::checkbox( "New application version", &newVersion );
        UI::setTooltipIfHovered( "Show when a new version of MeshInspector is available.", menuScaling );
        UI::checkbox( "Import warnings", &importWarnings );
        UI::setTooltipIfHovered( "Non-fatal warnings when importing a file", menuScaling );
    }
#ifndef __EMSCRIPTEN__
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Files and Folders" ) )
    {
        // TODO
        static std::string logFolderPath = Logger::instance().getLogFileName().parent_path().string();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 1.5f * cButtonPadding * menuScaling, cButtonPadding * menuScaling } );
        ImGui::InputText( "##LogFolderPath", logFolderPath, 0 );
        ImGui::SameLine( 0, 1.5f * style.ItemInnerSpacing.x );
        if ( ImGui::Link( "Logs folder") )
            OpenDocument( logFolderPath );
        ImGui::PopStyleVar();
        ImGui::SameLine( 0.0f, 0.0f );
        ImGui::SameLine( 0.0f, -30.0f * menuScaling );
        if ( UI::button( "...", ImVec2( 24.0f * menuScaling, 24.0f * menuScaling ) ) )
        {
            std::filesystem::path newPath = openFolderDialog( logFolderPath );
            if ( !newPath.empty() )
                logFolderPath = newPath.string();
        }
    }
#endif
}

void ViewerSettingsPlugin::drawControlTab_( float menuWidth, float menuScaling )
{
    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;

    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->setShowShortcuts( true );

    drawMouseSceneControlsSettings_( true, menuWidth, menuScaling );
    drawTouchpadSettings_();
    drawSpaceMouseSettings_( menuWidth, menuScaling );
}

void ViewerSettingsPlugin::drawViewportTab_( float menuWidth, float menuScaling )
{
    const auto& viewportParameters = viewer->viewport().getParameters();

    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Viewport", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        if ( viewer->viewport_list.size() > 1 )
            ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

        ImGui::SetNextItemWidth( 140.0f * menuScaling );
        auto rotMode = viewportParameters.rotationMode;
        UI::combo( "Rotation Mode", ( int* )&rotMode, { "Scene Center", "Pick / Scene Center", "Pick" } );
        viewer->viewport().rotationCenterMode( rotMode );

        ImGui::PushItemWidth( 80 * menuScaling );

        bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
        UI::checkbox( "Show Axes", &showAxes );
        viewer->viewport().showAxes( showAxes );

        bool showGlobalBasis = viewer->globalBasisAxes->isVisible( viewer->viewport().id );
        UI::checkbox( "Show Global Basis", &showGlobalBasis );
        viewer->viewport().showGlobalBasis( showGlobalBasis );

        bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
        UI::checkbox( "Show Rotation Center", &showRotCenter );
        viewer->viewport().showRotationCenter( showRotCenter );

        ImGui::PopItemWidth();

        drawBackgroundButton_( false );

        if ( viewer->isDeveloperFeaturesEnabled() &&
            RibbonButtonDrawer::CustomCollapsingHeader( "Clipping Plane" ) )
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
            UI::checkbox( "Show##ClippingPlane", &showPlane );
            viewer->viewport().setClippingPlane( plane );
            viewer->viewport().showClippingPlane( showPlane );
        }
    }
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::SetNextItemWidth( 100.0f * menuScaling );
        int pickRadius = int( getViewerInstance().glPickRadius );
        ImGui::DragInputInt( "Picker Radius", &pickRadius, 1, 0, 10 );
        getViewerInstance().glPickRadius = uint16_t( pickRadius );
        UI::setTooltipIfHovered( "Radius of area under cursor to pick objects in scene.", menuScaling );
    }
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Defaults", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        drawShadingModeCombo_( true, menuScaling );
        drawProjectionModeSelector_( true, menuScaling );
        drawUpDirectionSelector_();
    }
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Render", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        drawRenderOptions_( menuScaling );
        drawShadowsOptions_( menuWidth, menuScaling );
    }
}

void ViewerSettingsPlugin::drawRenderOptions_( float menuScaling )
{
    auto& style = ImGui::GetStyle();

    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Render Options" ) )
    {
        {
            if ( viewer->isAlphaSortAvailable() )
            {
                bool alphaSortBackUp = viewer->isAlphaSortEnabled();
                bool alphaBoxVal = alphaSortBackUp;
                UI::checkbox( "Alpha Sort", &alphaBoxVal );
                if ( alphaBoxVal != alphaSortBackUp )
                    viewer->enableAlphaSort( alphaBoxVal );
            }

            if ( viewer->isGLInitialized() )
            {
                if ( maxSamples_ > 1 )
                {
                    auto backUpSamples = storedSamples_;
                    ImGui::Text( "Multisample anti-aliasing (MSAA):" );
                    UI::setTooltipIfHovered( "The number of samples per pixel: more samples - better render quality but worse performance.", menuScaling );
                    int counter = 0;
                    for ( int i = 0; i <= maxSamples_; i <<= 1 )
                    {
                        if ( i == 0 )
                        {
                            UI::radioButton( "Off", &storedSamples_, i );
                            ++i;
                        }
                        else
                        {
                            std::string label = 'x' + std::to_string( i );
                            UI::radioButton( label.c_str(), &storedSamples_, i );
                        }
                        if ( i << 1 <= maxSamples_ )
                            ImGui::SameLine( ( ( ++counter ) * 70.f + style.WindowPadding.x ) * menuScaling );
                    }
                    if ( backUpSamples != storedSamples_ )
                    {
                        if ( auto& settingsManager = viewer->getViewportSettingsManager() )
                            settingsManager->saveInt( "multisampleAntiAliasing", storedSamples_ );

                        needReset_ = storedSamples_ != curSamples_;
                    }
                    if ( gpuOverridesMSAA_ )
                        UI::transparentTextWrapped( "GPU multisampling settings override application value." );
                    if ( needReset_ )
                        UI::transparentTextWrapped( "Application requires restart to apply this change" );
                }
            }
        }
    }
}

void ViewerSettingsPlugin::drawThemeSelector_( float menuWidth, float menuScaling )
{
    ImGui::SetNextItemWidth( menuWidth * 0.5f );
    int selectedUserIdxBackup = selectedUserPreset_;
    UI::combo( "Color theme", &selectedUserPreset_, userThemesPresets_ );
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
        ColorTheme::apply();
    }
    auto item = RibbonSchemaHolder::schema().items.find( "Add custom theme" );
    if ( item != RibbonSchemaHolder::schema().items.end() )
    {
        ImGui::SameLine( menuWidth * 0.75f );
        if ( UI::button( "Add",
            item->second.item->isAvailable( getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) ).empty(),
            Vector2f( menuWidth * 0.20f, 0 ) ) )
        {
            item->second.item->action();
        }
        UI::setTooltipIfHovered( item->second.tooltip, menuScaling );
    }
}

void ViewerSettingsPlugin::drawShadingModeCombo_( bool inGroup, float menuScaling )
{
    static std::vector<std::string> shadingModes = { "Auto Detect", "Smooth", "Flat" };
    SceneSettings::ShadingMode shadingMode = SceneSettings::getDefaultShadingMode();
    ImGui::SetNextItemWidth( 120.0f * menuScaling );
    UI::combo( inGroup ? "Shading Mode" : "Default Shading Mode", ( int* )&shadingMode, shadingModes);
    UI::setTooltipIfHovered( "Shading mode for mesh objects imported from files\n"
        "Detection depends on source format and mesh shape\n"
        "This setting also affects some tools", menuScaling );
    if ( shadingMode != SceneSettings::getDefaultShadingMode() )
        SceneSettings::setDefaultShadingMode( shadingMode );
}

void ViewerSettingsPlugin::drawProjectionModeSelector_( bool inGroup, float menuScaling )
{
    // TODO
    if ( !viewer->isDeveloperFeaturesEnabled() )
        return;
    ImGui::SetNextItemWidth( 120.0f * menuScaling );
    static std::vector<std::string> projectionModes = { "Orthogonal", "Perspective" };
    static int projectionMode = 0;
    UI::combo( inGroup ? "Projection Mode" : "Default Projection Mode", ( int* )&projectionMode, projectionModes);
}

void ViewerSettingsPlugin::drawUpDirectionSelector_()
{
    // TODO
    if ( !viewer->isDeveloperFeaturesEnabled() )
        return;
    ImGui::Text( "Up Direction" );
    static int axis = 2; // Z
    ImGui::SameLine();
    UI::radioButton( "Y", &axis, 1 );
    ImGui::SameLine();
    UI::radioButton( "Z", &axis, 2 );
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

void ViewerSettingsPlugin::drawShadowsOptions_( float menuWidth, float menuScaling )
{
    if ( shadowGl_ && RibbonButtonDrawer::CustomCollapsingHeader( "Shadows" ) )
    {
        auto& style = ImGui::GetStyle();
        bool isEnableShadows = shadowGl_->isEnabled();
        UI::checkbox( "Enabled", &isEnableShadows );
        if ( isEnableShadows != shadowGl_->isEnabled() )
        {
            CommandLoop::appendCommand( [shadowGl = shadowGl_.get(), isEnableShadows] ()
            {
                shadowGl->enable( isEnableShadows );
            } );
        }
        ImGui::SameLine( menuWidth * 0.25f + style.WindowPadding.x + 2 * menuScaling );
        auto color = shadowGl_->getShadowColor();
        UI::colorEdit4( "Shadow Color", color,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
        shadowGl_->setShadowColor( color );

        const char* tooltipsShift[2] = {
            "Shift along Ox-axis to the left",
            "Shift along Oy-axis to the top"
        };
        ImGui::PushItemWidth( menuWidth * 0.5f );
        auto shfit = shadowGl_->getShadowShift();
        auto radius = shadowGl_->getBlurRadius();
        auto quality = shadowGl_->getQuality();
        ImGui::DragFloatValid2( "Shift", &shfit.x, 0.4f, -200.0f, 200.0f, "%.3f px", 0, &tooltipsShift );
        ImGui::DragFloatValid( "Blur Radius", &radius, 0.2f, 0, 200, "%.3f px" );
        ImGui::DragFloatValid( "Quality", &quality, 0.001f, 0.0625f, 1.0f );
        ImGui::PopItemWidth();
        UI::setTooltipIfHovered( "Blur texture downscaling coefficient", menuScaling );
        shadowGl_->setShadowShift( shfit );
        shadowGl_->setBlurRadius( radius );
        shadowGl_->setQuality( quality );
    }
}

void ViewerSettingsPlugin::drawMouseSceneControlsSettings_( bool defaultOpen, float menuWidth, float scaling )
{
    if ( !ImGui::CollapsingHeader( "Mouse Control", defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0 ) )
        return;

    const float buttonHeight = cGradientButtonFramePadding * scaling + ImGui::CalcTextSize( "Set other" ).y;
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
        ImGui::SetCursorPosY( posY + cGradientButtonFramePadding * scaling / 2.f );
        ImGui::Text( "%s", modeName.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( menuWidth * 0.5f - 50.0f * scaling );
        ImGui::Text( "%s", ctrlStr.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( menuWidth - 150.0f * scaling );

		ImGui::SetCursorPosY( posY );
        UI::button( "Set other", Vector2f( -1, buttonHeight ) );
        if ( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ImGui::Text( "Click here with preferred mouse button \nwith/without modifier (alt/ctrl/shift)" );
            ImGui::EndTooltip();
            
            if ( ImGui::GetIO().MouseClicked[0] || ImGui::GetIO().MouseClicked[1] || ImGui::GetIO().MouseClicked[2] )
            {
                MouseButton clikedBtn = MouseButton::Left;
                if ( ImGui::GetIO().MouseClicked[1] )
                    clikedBtn = MouseButton::Right;
                else if ( ImGui::GetIO().MouseClicked[2] )
                    clikedBtn = MouseButton::Middle;

                int modifier = 0;
                if ( ImGui::GetIO().KeyCtrl )
                    modifier |= GLFW_MOD_CONTROL;
                if ( ImGui::GetIO().KeyAlt )
                    modifier |= GLFW_MOD_ALT;
                if ( ImGui::GetIO().KeyShift )
                    modifier |= GLFW_MOD_SHIFT;

                viewer->mouseController().setMouseControl( { clikedBtn,modifier }, mode );
            }
        }
    }

    ImGui::SetNextItemWidth( 100 * scaling );
    ImGui::DragFloatValid( "Scroll modifier", &viewer->scrollForce, 0.01f, 0.2f, 3.0f );
}

void ViewerSettingsPlugin::drawSpaceMouseSettings_( float menuWidth, float scaling )
{
    if ( !ImGui::CollapsingHeader( "Spacemouse Settings" ) )
        return;

    bool anyChanged = false;
    auto drawSlider = [&anyChanged, menuWidth] ( const char* label, float& value )
    {
        int valueAbs = int( std::fabs( value ) );
        bool inverse = value < 0.f;
        ImGui::SetNextItemWidth( menuWidth * 0.6f );
        bool changed = UI::sliderInt( label, &valueAbs, 1, 100 );
        ImGui::SameLine( menuWidth * 0.78f );
        const float cursorPosY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( cursorPosY + 3 );
        changed = UI::checkbox( ( std::string( "Inverse##" ) + label ).c_str(), &inverse ) || changed;
        if ( changed )
            value = valueAbs * ( inverse ? -1.f : 1.f );
        anyChanged = anyChanged || changed;
    };

    ImGui::Text( "%s", "Translation scales" );
    drawSlider( "X##translate", spaceMouseParams_.translateScale[0] );
    drawSlider( "Y##translate", spaceMouseParams_.translateScale[2] );
    drawSlider( "Zoom##translate", spaceMouseParams_.translateScale[1] );

    ImGui::Text( "%s", "Rotation scales" );
    drawSlider( "Ox##rotate", spaceMouseParams_.rotateScale[0] );
    drawSlider( "Oy##rotate", spaceMouseParams_.rotateScale[1] );
    drawSlider( "Oz##rotate", spaceMouseParams_.rotateScale[2] );
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
    UI::setTooltipIfHovered( "This mode is NOT recommended if you have 3Dconnexion driver installed, which sends mouse wheel fake events resulting in double reaction on SpaceMouse movement and camera tremble.", scaling );
#else
    (void)scaling;
#endif
    if ( anyChanged )
        getViewerInstance().setSpaceMouseParameters( spaceMouseParams_ );
}

void ViewerSettingsPlugin::drawTouchpadSettings_()
{
    if ( !ImGui::CollapsingHeader( "Touchpad Settings" ) )
        return;

    const std::vector<std::string> swipeModeList = { "Swipe Rotates Camera", "Swipe Moves Camera" };
    assert( swipeModeList.size() == (size_t)TouchpadParameters::SwipeMode::Count );

    bool updateSettings = false;
    if ( UI::checkbox( "Ignore Kinetic Movements", &touchpadParameters_.ignoreKineticMoves ) )
        updateSettings = true;
    if ( UI::checkbox( "Allow System to Interrupt Gestures", &touchpadParameters_.cancellable ) )
        updateSettings = true;
    if ( UI::combo( "Swipe Mode", (int*)&touchpadParameters_.swipeMode, swipeModeList ) )
        updateSettings = true;
    if ( updateSettings )
        viewer->setTouchpadParameters( touchpadParameters_ );
}

void ViewerSettingsPlugin::drawCustomSettings_( TabType tabType, float scaling )
{
    if ( comboSettings_[size_t( tabType )].empty() )
        return;
    ImGui::Separator();
    for ( auto& settings : comboSettings_[size_t( tabType )] )
    {
        settings->draw( scaling );
    }
}

MR_REGISTER_RIBBON_ITEM( ViewerSettingsPlugin )

}
