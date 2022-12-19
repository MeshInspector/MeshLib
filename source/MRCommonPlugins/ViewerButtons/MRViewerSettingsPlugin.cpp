#include "MRViewerSettingsPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRColorTheme.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewerSettingsManager.h"
#include "MRViewer/MRGLMacro.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonConstants.h"

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
}

void ViewerSettingsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 300.0f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    if ( RibbonButtonDrawer::GradientButton( "Quick Access Menu Settings", ImVec2( -1, 0 ) ) )
        ImGui::OpenPopup( "Quick Access Menu Settings" );
    drawDialogQuickAccessSettings_( menuScaling );

    if ( RibbonButtonDrawer::GradientButton( "Scene mouse controls", ImVec2( -1, 0 ) ) )
        ImGui::OpenPopup( "Scene mouse controls" );
    drawMouseSceneControlsSettings_( menuScaling );

    if ( RibbonButtonDrawer::GradientButton( "Show hotkeys", ImVec2( -1, 0 ) ) && ribbonMenu_ )
    {
        ribbonMenu_->setShowShortcuts( true );
    }
    if ( RibbonButtonDrawer::GradientButton( "Spacemouse settings", ImVec2( -1, 0 ) ) )
    {
        spaceMouseParams = getViewerInstance().spaceMouseController.getParams();
        ImGui::OpenPopup( "Spacemouse settings" );
    }
    drawSpaceMouseSettings_( menuScaling );

    const auto& viewportParameters = viewer->viewport().getParameters();
    // Viewing options
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Current viewport options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

        ImGui::PushItemWidth( 80 * menuScaling );
        bool showGlobalBasis = viewer->globalBasisAxes->isVisible( viewer->viewport().id );
        RibbonButtonDrawer::GradientCheckbox( "Show Global Basis", &showGlobalBasis );
        viewer->viewport().showGlobalBasis( showGlobalBasis );

        bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
        RibbonButtonDrawer::GradientCheckbox( "Show rotation center", &showRotCenter );
        viewer->viewport().showRotationCenter( showRotCenter );

        ImGui::SetNextItemWidth( 140.0f * menuScaling );
        auto rotMode = viewportParameters.rotationMode;
        RibbonButtonDrawer::CustomCombo( "Rotation Mode", ( int* )&rotMode, { "Scene Center", "Pick / Scene Center", "Pick" } );
        viewer->viewport().rotationCenterMode( rotMode );

        bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
        RibbonButtonDrawer::GradientCheckbox( "Show axes", &showAxes );
        viewer->viewport().showAxes( showAxes );
        ImGui::PopItemWidth();


        bool needUpdateBackup = backgroundColor_.w == -1.0f;
        if ( needUpdateBackup )
            backgroundColor_ = Vector4f( viewportParameters.backgroundColor );

        auto backgroundColor = backgroundColor_;
        if ( ImGui::ColorEdit4( "Background", &backgroundColor.x,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
            backgroundColor_ = backgroundColor;
        else if ( !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) )
            backgroundColor_.w = -1.0f;
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );

        if ( viewer->isDeveloperFeaturesEnabled() &&
            RibbonButtonDrawer::CustomCollapsingHeader( "Clipping plane" ) )
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
            RibbonButtonDrawer::GradientCheckbox( "Show##ClippingPlane", &showPlane );
            viewer->viewport().setClippingPlane( plane );
            viewer->viewport().showClippingPlane( showPlane );
        }
    }
    ImGui::Separator();
    if ( RibbonButtonDrawer::CustomCollapsingHeader( "Viewer options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::Separator();
        ImGui::SetNextItemWidth( 125.0f * menuScaling );
        int selectedUserIdxBackup = selectedUserPreset_;
        RibbonButtonDrawer::CustomCombo( "Color theme", &selectedUserPreset_, userThemesPresets_ );
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

                if ( auto menu = getViewerInstance().getMenuPlugin() )
                    menu->showErrorModal( "This theme is not valid." );
            }
            backgroundColor_ = Vector4f( ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::Background ) );
            ColorTheme::apply();
        }
        auto item = RibbonSchemaHolder::schema().items.find( "Add custom theme" );
        if ( item != RibbonSchemaHolder::schema().items.end() )
        {
            ImGui::SameLine();
            if ( RibbonButtonDrawer::GradientButtonValid( "Add",
                item->second.item->isAvailable(
                    getAllObjectsInTree<const Object>( &SceneRoot::get(),
                        ObjectSelectivityType::Selected ) ).empty(),
                ImVec2( -1, ImGui::GetFrameHeight() ) ) )
            {
                item->second.item->action();
            }
            ImGui::SetTooltipIfHovered( item->second.tooltip, menuScaling );
        }

        ImGui::Separator();

        if ( ribbonMenu_ )
        {
            RibbonButtonDrawer::GradientCheckbox( "Make visible on select",
                                                  std::bind( &RibbonMenu::getShowNewSelectedObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setShowNewSelectedObjects, ribbonMenu_, std::placeholders::_1 ) );
            RibbonButtonDrawer::GradientCheckbox( "Deselect on hide",
                                                  std::bind( &RibbonMenu::getDeselectNewHiddenObjects, ribbonMenu_ ),
                                                  std::bind( &RibbonMenu::setDeselectNewHiddenObjects, ribbonMenu_, std::placeholders::_1 ) );
        }

        bool flatShading = SceneSettings::get( SceneSettings::Type::MeshFlatShading );
        bool flatShadingBackup = flatShading;
        RibbonButtonDrawer::GradientCheckbox( "Default shading flat", &flatShading );
        if ( flatShadingBackup != flatShading )
            SceneSettings::set( SceneSettings::Type::MeshFlatShading, flatShading );

        if ( viewer->isAlphaSortAvailable() )
        {
            bool alphaSortBackUp = viewer->isAlphaSortEnabled();
            bool alphaBoxVal = alphaSortBackUp;
            RibbonButtonDrawer::GradientCheckbox( "Alpha Sort", &alphaBoxVal );
            if ( alphaBoxVal != alphaSortBackUp )
                viewer->enableAlphaSort( alphaBoxVal );
        }
        if ( viewer->isGLInitialized() )
        {
            if ( maxSamples_ > 1 )
            {
                auto backUpSamples = storedSamples_;
                ImGui::Text( "Multisample anti-aliasing (MSAA):" );
                ImGui::SetTooltipIfHovered( "The number of samples per pixel: more samples - better render quality but worse performance.", menuScaling );
                int couter = 0;
                for ( int i = 0; i <= maxSamples_; i <<= 1 )
                {
                    if ( i == 0 )
                    {
                        RibbonButtonDrawer::GradientRadioButton( "Off", &storedSamples_, i );
                        ++i;
                    }
                    else
                    {
                        std::string label = 'x' + std::to_string( i );
                        RibbonButtonDrawer::GradientRadioButton( label.c_str(), &storedSamples_, i );
                    }
                    if ( i << 1 <= maxSamples_ && ( ++couter ) % 3 != 0 )
                        ImGui::SameLine( ( couter % 3 ) * menuScaling * 80.0f );
                }
                if ( backUpSamples != storedSamples_ )
                {
                    if ( auto& settingsManager = viewer->getViewportSettingsManager() )
                        settingsManager->saveInt( "multisampleAntiAliasing", storedSamples_ );
                    
                    needReset_ = storedSamples_ != curSamples_;
                }
                if ( needReset_ )
                    ImGui::TransparentTextWrapped( "Application requires restart to apply this change" );
            }
        }
        if ( shadowGl_ && RibbonButtonDrawer::CustomCollapsingHeader( "Shadows" ) )
        {
            bool isEnableShadows = shadowGl_->isEnabled();
            RibbonButtonDrawer::GradientCheckbox( "Enabled", &isEnableShadows );
            if ( isEnableShadows != shadowGl_->isEnabled() )
            {
                CommandLoop::appendCommand( [shadowGl = shadowGl_.get(), isEnableShadows] ()
                {
                    shadowGl->enable( isEnableShadows );
                } );
            }
            ImGui::ColorEdit4( "Color", &shadowGl_->shadowColor.x,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
            
            const char* tooltipsShift[2] = {
                "Shift along Ox-axis to the left",
                "Shift along Oy-axis to the top"
            };
            ImGui::DragFloatValid2( "Shift", &shadowGl_->shadowShift.x, 0.4f, -200.0f, 200.0f, "%.3f px", 0, &tooltipsShift );
            ImGui::DragFloatValid( "Blur radius", &shadowGl_->blurRadius, 0.2f, 0, 200, "%.3f px" );
            float quality = shadowGl_->getQuality();
            ImGui::DragFloatValid( "Quality", &quality, 0.001f, 0.0625f, 1.0f );
            ImGui::SetTooltipIfHovered( "Blur texture downscaling coefficient", menuScaling );
            shadowGl_->setQuality( quality );
        }
    }
    ImGui::EndCustomStatePlugin();
}

bool ViewerSettingsPlugin::onEnable_()
{
    backgroundColor_.w = -1.0f;

    ribbonMenu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>().get();
    if ( ribbonMenu_ )
    {
        schema_ = &RibbonSchemaHolder::schema();
        quickAccessList_ = &ribbonMenu_->getQuickAccessList();
        maxQuickAccessSize_ = ribbonMenu_->getQuickAccessMaxSize();
    }

    selectedUserPreset_ = -1;
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
        for ( const auto& entry : std::filesystem::directory_iterator( userThemesDir, ec ) )
        {
            if ( entry.is_regular_file( ec ) )
            {
                auto ext = entry.path().extension().u8string();
                for ( auto& c : ext )
                    c = ( char ) tolower( c );

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

    return true;
}

bool ViewerSettingsPlugin::onDisable_()
{
    userThemesPresets_.clear();
    schema_ = nullptr;
    quickAccessList_ = nullptr;
    ribbonMenu_ = nullptr;
    return true;
}


void ViewerSettingsPlugin::drawMouseSceneControlsSettings_( float scaling )
{
    auto& viewerRef = Viewer::instanceRef();
    ImVec2 windowSize( 500 * scaling, 150 * scaling );
    ImGui::SetNextWindowPos( ImVec2( ( viewerRef.window_width - windowSize.x ) / 2.f, ( viewerRef.window_height - windowSize.y ) / 2.f ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    if ( !ImGui::BeginModalNoAnimation( "Scene mouse controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
        return;

    ImGui::Text( "%s", "Mouse scene controls" );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 20.f );

    drawModalExitButton_( scaling );

    ImGui::BeginChild( "##MouseSceneControlsList", ImVec2( -1, -1 ), true );

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
        ImGui::SetCursorPosX( windowSize.x * 0.5f - 50.0f );
        ImGui::Text( "%s", ctrlStr.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( windowSize.x - 150.0f );

		ImGui::SetCursorPosY( posY );
        RibbonButtonDrawer::GradientButton( "Set other", ImVec2( -1, buttonHeight ) );
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

    ImGui::DragFloatValid( "Scroll modifier", &viewer->scrollForce, 0.01f, 0.2f, 3.0f );

    ImGui::EndChild();

    ImGui::EndPopup();
}

void ViewerSettingsPlugin::drawDialogQuickAccessSettings_( float scaling )
{
    auto& viewerRef = Viewer::instanceRef();
    ImVec2 windowSize( 500 * scaling, 400 * scaling );
    ImGui::SetNextWindowPos( ImVec2( ( viewerRef.window_width - windowSize.x ) / 2.f, ( viewerRef.window_height - windowSize.y ) / 2.f ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    if ( !ImGui::BeginModalNoAnimation( "Quick Access Menu Settings", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
        return;

    ImGui::Text( "%s", "Toolbar Settings" );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 20.f );

    drawModalExitButton_( scaling );

    ImGui::Text( "%s", "Select icons to show in Toolbar" );

    ImGui::SameLine();
    auto& style = ImGui::GetStyle();
    float textPosX = windowSize.x - ImGui::CalcTextSize( "Icons in Toolbar : 00/00" ).x - style.WindowPadding.x;
    ImGui::SetCursorPosX( textPosX );
    ImGui::Text( "Icons in Toolbar : %02d/%02d", int( quickAccessList_->size() ), maxQuickAccessSize_ );
    
    const float buttonHeight = cGradientButtonFramePadding * scaling + ImGui::CalcTextSize( "Reset to default" ).y;
    const float height = ImGui::GetStyle().ItemSpacing.y + buttonHeight;

    ImGui::BeginChild( "##QuickAccessSettingsList", ImVec2( -1, -height ), true );

    drawQuickAccessList_();

    ImGui::EndChild();

    if ( RibbonButtonDrawer::GradientButton( "Reset to default", ImVec2( 0, buttonHeight ) ) )
    {
        auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
        if ( ribbonMenu )
            ribbonMenu->resetQuickAccessList();
    }

    ImGui::EndPopup();
}

void ViewerSettingsPlugin::drawQuickAccessList_()
{
    auto& tabsOrder = schema_->tabsOrder;
    auto& tabsMap = schema_->tabsMap;
    auto& groupsMap = schema_->groupsMap;
    auto& quickAccessList = *quickAccessList_;

    bool canAdd = int( quickAccessList.size() ) < maxQuickAccessSize_;

    int quickAccessListIndex = 0;
    for ( const auto& [tabName, tabPriority]  : tabsOrder )
    {
        if ( !ImGui::TreeNodeEx( tabName.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen ) )
            continue;
        auto tabIt = tabsMap.find( tabName );
        if ( tabIt == tabsMap.end() )
            continue;
        auto& tab = tabIt->second;
        for ( auto& group : tab )
        {
            if ( !ImGui::TreeNodeEx( group.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen ) )
                continue;
            auto itemsIt = groupsMap.find( tabName + group );
            if ( itemsIt == groupsMap.end() )
                continue;
            auto& items = itemsIt->second;
            for ( auto& item : items )
            {
                if ( item == "Quick Access Settings" )
                    continue;
                if ( quickAccessListIndex < quickAccessList.size() && quickAccessList[quickAccessListIndex] == item )
                    ++quickAccessListIndex;

                auto itemIt = std::find( quickAccessList.begin(), quickAccessList.end(), item );
                bool itemInQA = itemIt != quickAccessList.end();

                bool disabled = !canAdd && !itemInQA;
                if ( disabled )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
                    ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ImGui::GetColorU32( ImGuiCol_FrameBg ) );
                    ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImGui::GetColorU32( ImGuiCol_FrameBg ) );
                }

                if ( RibbonButtonDrawer::GradientCheckbox( item.c_str(), &itemInQA ) )
                {
                    if ( itemInQA )
                    {
                        if ( canAdd )
                            quickAccessList.emplace( quickAccessList.begin() + quickAccessListIndex, item );
                        else
                            itemInQA = false;
                    }
                    else
                        quickAccessList.erase( itemIt );
                }

                if ( disabled )
                    ImGui::PopStyleColor( 3 );


            }
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
}

void ViewerSettingsPlugin::drawSpaceMouseSettings_( float scaling )
{
    auto& viewerRef = Viewer::instanceRef();
    ImVec2 windowSize( 400 * scaling, 285 * scaling );
    ImGui::SetNextWindowPos( ImVec2( ( viewerRef.window_width - windowSize.x ) / 2.f, ( viewerRef.window_height - windowSize.y ) / 2.f ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    if ( !ImGui::BeginModalNoAnimation( "Spacemouse settings", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
        return;

    ImGui::Text( "%s", "Spacemouse settings" );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 20.f );

    drawModalExitButton_( scaling );

    ImGui::Text( "%s", "Translation scales" );

    bool anyChanged = false;
    auto drawSlider = [&anyChanged, &windowSize] ( const char* label, float& value )
    {
        int valueAbs = int( std::fabs( value ) );
        bool inverse = value < 0.f;
        bool changed = ImGui::SliderInt( label, &valueAbs, 1, 100 );
        ImGui::SameLine( windowSize.x * 0.8f);
        const float cursorPosY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( cursorPosY + 3 );
        changed = RibbonButtonDrawer::GradientCheckbox( ( std::string( "Inverse##" ) + label ).c_str(), &inverse ) || changed;
        if ( changed )
            value = valueAbs * ( inverse ? -1.f : 1.f );
        anyChanged = anyChanged || changed;
    };

    drawSlider( "X##translate", spaceMouseParams.translateScale[0] );
    drawSlider( "Y##translate", spaceMouseParams.translateScale[2] );
    drawSlider( "Zoom##translate", spaceMouseParams.translateScale[1] );

    ImGui::NewLine();
    ImGui::Text( "%s", "Rotation scales" );
    drawSlider( "Ox##rotate", spaceMouseParams.rotateScale[0] );
    drawSlider( "Oy##rotate", spaceMouseParams.rotateScale[1] );
    drawSlider( "Oz##rotate", spaceMouseParams.rotateScale[2] );

    if ( anyChanged )
        getViewerInstance().spaceMouseController.setParams( spaceMouseParams );

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
    if ( RibbonButtonDrawer::GradientButton( "\xef\x80\x8d", ImVec2( btnSize , btnSize ) ) )
        ImGui::CloseCurrentPopup();
    ImGui::PopFont();
    ImGui::SetCursorPos( oldCursorPos );
}

MR_REGISTER_RIBBON_ITEM( ViewerSettingsPlugin )

}