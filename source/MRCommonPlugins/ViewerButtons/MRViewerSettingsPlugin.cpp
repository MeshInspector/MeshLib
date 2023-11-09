#include "MRViewerSettingsPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRColorTheme.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewerSettingsManager.h"
#include "MRViewer/MRGLMacro.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRSpaceMouseHandlerHidapi.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRUIStyle.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRDirectory.h"


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

void ViewerSettingsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 380.0f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;


    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, MR::StyleConsts::pluginItemSpacing );

    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;

    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->openToolbarCustomize();

    ImGui::SameLine();
    if ( UI::button( "Scene Mouse Controls", Vector2f( btnHalfSizeX, 0 ) ) )
        ImGui::OpenPopup( "Scene Mouse Controls" );
    drawMouseSceneControlsSettings_( menuScaling );

    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->setShowShortcuts( true );

    ImGui::SameLine();
    if ( UI::button( "Spacemouse Settings", Vector2f( btnHalfSizeX, 0 ) ) )
    {
        auto& viewerRef = getViewerInstance();
        spaceMouseParams_ = viewerRef.spaceMouseController.getParams();
#if defined(_WIN32) || defined(__APPLE__)
        if ( auto spaceMouseHandler = viewerRef.getSpaceMouseHandler() )
        {
            auto hidapiHandler = std::dynamic_pointer_cast<MR::SpaceMouseHandlerHidapi>( spaceMouseHandler );
            if ( hidapiHandler )
                activeMouseScrollZoom_ = hidapiHandler->isMouseScrollZoomActive();
        }
#endif
        ImGui::OpenPopup( "Spacemouse Settings" );
    }
    drawSpaceMouseSettings_( menuScaling );

    if ( UI::button( "Touchpad Settings", Vector2f( btnHalfSizeX, 0 ) ) )
    {
        touchpadParameters_ = viewer->touchpadController.getParameters();
        ImGui::OpenPopup( "Touchpad Settings" );
    }
    drawTouchpadSettings_( menuScaling );

    const auto& viewportParameters = viewer->viewport().getParameters();
    // Viewing options
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Current Viewport Options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        //ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

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

        
        bool needUpdateBackup = backgroundColor_.w == -1.0f;
        if ( needUpdateBackup )
            backgroundColor_ = Vector4f( viewportParameters.backgroundColor );

        auto backgroundColor = backgroundColor_;

        ImGui::SameLine( menuWidth * 0.5f );
        if ( UI::colorEdit4( "Background Color", backgroundColor,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
            backgroundColor_ = backgroundColor;
        else if ( !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) )
            backgroundColor_.w = -1.0f;
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );

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
    ImGui::Separator();
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Viewer Options", ImGuiTreeNodeFlags_DefaultOpen ) )
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

        ImGui::Separator();

        if ( ribbonMenu_ )
        {
            UI::checkbox( "Make Visible on Select",
                                                  std::bind( &RibbonMenu::getShowNewSelectedObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setShowNewSelectedObjects, ribbonMenu_, std::placeholders::_1 ) );
            UI::checkbox( "Deselect on Hide",
                                                  std::bind( &RibbonMenu::getDeselectNewHiddenObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setDeselectNewHiddenObjects, ribbonMenu_, std::placeholders::_1 ) );
        }

        bool flatShading = SceneSettings::get( SceneSettings::Type::MeshFlatShading );
        bool flatShadingBackup = flatShading;
        UI::checkbox( "Default Shading Flat", &flatShading );
        if ( flatShadingBackup != flatShading )
            SceneSettings::set( SceneSettings::Type::MeshFlatShading, flatShading );

        ImGui::SetNextItemWidth( 100.0f * menuScaling );
        int pickRadius = int( getViewerInstance().glPickRadius );
        ImGui::DragInputInt( "Picker Radius", &pickRadius, 1, 0, 10 );
        getViewerInstance().glPickRadius = uint16_t( pickRadius );
        UI::setTooltipIfHovered( "Radius of area under cursor to pick objects in scene.", menuScaling );

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
                ImGui::Separator();
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
        if ( shadowGl_ && RibbonButtonDrawer::CustomCollapsingHeader( "Shadows" ) )
        {
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

    ImGui::PopStyleVar();

    ImGui::EndCustomStatePlugin();
}

bool ViewerSettingsPlugin::onEnable_()
{
    backgroundColor_.w = -1.0f;

    ribbonMenu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>().get();
    updateThemes();

    return true;
}

bool ViewerSettingsPlugin::onDisable_()
{
    userThemesPresets_.clear();
    ribbonMenu_ = nullptr;
    return true;
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

void ViewerSettingsPlugin::drawMouseSceneControlsSettings_( float scaling )
{
    ImVec2 windowSize = ImVec2( 500 * scaling, 0 );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos( center, ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( windowSize.x, -1 ), ImVec2( windowSize.x, 0 ) );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 3 * MR::cDefaultItemSpacing * scaling, 3 * MR::cDefaultItemSpacing * scaling ) );
    if ( !ImGui::BeginModalNoAnimation( "Scene Mouse Controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::PopStyleVar();
        return;
    }
    ImGui::PopStyleVar();

    if ( ImGui::ModalBigTitle( "Scene Mouse Controls", scaling ) )
    {
        ImGui::CloseCurrentPopup();
    }

	const float buttonHeight = cGradientButtonFramePadding * scaling + ImGui::CalcTextSize( "Set other" ).y;
    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        if ( mode == MouseMode::None )
            continue;
        std::string modeName = getMouseModeString( mode );
        std::string ctrlStr = "None";
        auto ctrl = viewer->mouseController.findControlByMode( mode );
        if ( ctrl )
            ctrlStr = MouseController::getControlString( *ctrl );

        const float posY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( posY + cGradientButtonFramePadding * scaling / 2.f );
        ImGui::Text( "%s", modeName.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( windowSize.x * 0.5f - 50.0f * scaling );
        ImGui::Text( "%s", ctrlStr.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( windowSize.x - 150.0f * scaling );

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

                viewer->mouseController.setMouseControl( { clikedBtn,modifier }, mode );
            }
        }
    }

    ImGui::SetNextItemWidth( 100 * scaling );
    ImGui::DragFloatValid( "Scroll modifier", &viewer->scrollForce, 0.01f, 0.2f, 3.0f );

    ImGui::EndPopup();
}

void ViewerSettingsPlugin::drawSpaceMouseSettings_( float scaling )
{
    ImVec2 windowSize = ImVec2( 450 * scaling, 0);
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImGui::SetNextWindowSizeConstraints( ImVec2( windowSize.x, -1 ), ImVec2( windowSize.x, 0 ) );

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos( center, ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 3 * MR::cDefaultItemSpacing * scaling, 3 * MR::cDefaultItemSpacing * scaling ) );
    if ( !ImGui::BeginModalNoAnimation( "Spacemouse Settings", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::PopStyleVar();
        return;
    }

    if ( ImGui::ModalBigTitle( "Spacemouse Settings", scaling ) )
        ImGui::CloseCurrentPopup();


    auto font = RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::BigSemiBold );
    if ( font )
        ImGui::PushFont( font );
    ImGui::Text( "%s", "Translation scales" );
    if ( font )
        ImGui::PopFont();

    bool anyChanged = false;
    auto drawSlider = [&anyChanged, &windowSize] ( const char* label, float& value )
    {
        int valueAbs = int( std::fabs( value ) );
        bool inverse = value < 0.f;
        ImGui::SetNextItemWidth( windowSize.x * 0.6f );
        bool changed = ImGui::SliderInt( label, &valueAbs, 1, 100 );
        ImGui::SameLine( windowSize.x * 0.78f );
        const float cursorPosY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( cursorPosY + 3 );
        changed = UI::checkbox( ( std::string( "Inverse##" ) + label ).c_str(), &inverse ) || changed;
        if ( changed )
            value = valueAbs * ( inverse ? -1.f : 1.f );
        anyChanged = anyChanged || changed;
    };

    drawSlider( "X##translate", spaceMouseParams_.translateScale[0] );
    drawSlider( "Y##translate", spaceMouseParams_.translateScale[2] );
    drawSlider( "Zoom##translate", spaceMouseParams_.translateScale[1] );

    ImGui::NewLine();
    if ( font )
        ImGui::PushFont( font );
    ImGui::Text( "%s", "Rotation scales" );
    if ( font )
        ImGui::PopFont();
    drawSlider( "Ox##rotate", spaceMouseParams_.rotateScale[0] );
    drawSlider( "Oy##rotate", spaceMouseParams_.rotateScale[1] );
    drawSlider( "Oz##rotate", spaceMouseParams_.rotateScale[2] );
#if defined(_WIN32) || defined(__APPLE__)
    ImGui::NewLine();
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
#endif
    if ( anyChanged )
        getViewerInstance().spaceMouseController.setParams( spaceMouseParams_ );

    ImGui::PopStyleVar();
    ImGui::EndPopup();
}

void ViewerSettingsPlugin::drawTouchpadSettings_( float menuScaling )
{
    ImVec2 windowSize = ImVec2( 500 * menuScaling, 0 );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos( center, ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( windowSize.x, -1 ), ImVec2( windowSize.x, 0 ) );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 3 * MR::cDefaultItemSpacing * menuScaling, 3 * MR::cDefaultItemSpacing * menuScaling ) );
    if ( !ImGui::BeginModalNoAnimation( "Touchpad Settings", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::PopStyleVar();
        return;
    }
    ImGui::PopStyleVar();

    if ( ImGui::ModalBigTitle( "Touchpad Settings", menuScaling ) )
    {
        ImGui::CloseCurrentPopup();
    }

    const std::vector<std::string> swipeModeList = { "Swipe Rotates Camera", "Swipe Moves Camera" };
    assert( swipeModeList.size() == TouchpadController::Parameters::SwipeModeCount );

    bool updateSettings = false;
    if ( UI::checkbox( "Ignore Kinetic Movements", &touchpadParameters_.ignoreKineticMoves ) )
        updateSettings = true;
    if ( UI::checkbox( "Allow System to Interrupt Gestures", &touchpadParameters_.cancellable ) )
        updateSettings = true;
    if ( UI::combo( "Swipe Mode", (int*)&touchpadParameters_.swipeMode, swipeModeList ) )
        updateSettings = true;
    if ( updateSettings )
        viewer->touchpadController.setParameters( touchpadParameters_ );

    ImGui::EndPopup();
}

void ViewerSettingsPlugin::drawModalExitButton_( float scaling )
{
    ImVec2 oldCursorPos = ImGui::GetCursorPos();
    ImVec2 windowSize = ImGui::GetWindowSize();
    ImVec2 btnPos = ImVec2( windowSize.x - 30 * scaling, 10 * scaling );
    ImGui::SetCursorPos( btnPos );
    auto font = RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::Icons );
    if ( !font )
        return;
    font->Scale = 0.6f;
    ImGui::PushFont( font );
    const float btnSize = 20 * scaling;
    if ( UI::button( "\xef\x80\x8d", Vector2f( btnSize , btnSize ) ) )
        ImGui::CloseCurrentPopup();
    ImGui::PopFont();
    ImGui::SetCursorPos( oldCursorPos );
}

MR_REGISTER_RIBBON_ITEM( ViewerSettingsPlugin )

}