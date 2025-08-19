#include "MRAddCustomTheme.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRShowModal.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRMesh/MRSceneColors.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRModalDialog.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRViewer/MRRibbonButtonDrawer.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/MRViewerSettingsPlugin.h"
#include "MRViewer/MRUIStyle.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRPch/MRJson.h"
#include <fstream>

#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif

namespace MR
{

AddCustomThemePlugin::AddCustomThemePlugin():
    StatePlugin( "Add custom theme" )
{
}

void AddCustomThemePlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 450.0f * menuScaling;
    auto menuHeight = 600.0f * menuScaling;

    if ( !ImGuiBeginWindow_( { .width = menuWidth,.height = menuHeight, .menuScaling = menuScaling, .flags = 0 } ) )
        return;

    int selectedUserIdxBackup = selectedUserPreset_;
    ImGui::PushItemWidth( 220.0f * menuScaling );
    UI::combo( "Ribbon theme preset", &selectedUserPreset_, userThemesPresets_ );
    if ( selectedUserPreset_ != selectedUserIdxBackup )
        update_();
    ImGui::Separator();

    ImGui::Text( "Scene colors:" );
    for ( int i = 0; i < sceneColors_.size(); ++i )
        ImGui::ColorEdit4( SceneColors::getName( SceneColors::Type( i ) ), &sceneColors_[i].x );
    ImGui::Separator();

    ImGui::Text( "UI colors:" );
    for ( int i = 0; i < ribbonColors_.size(); ++i )
        ImGui::ColorEdit4( ColorTheme::getRibbonColorTypeName( ColorTheme::RibbonColorsType( i ) ), &ribbonColors_[i].x );
    ImGui::Separator();

    ImGui::Text( "Viewport colors:" );
    for ( int i = 0; i < viewportColors_.size(); ++i )
    {
        std::string label = ColorTheme::getViewportColorTypeName( ColorTheme::ViewportColorsType( i ) ) +
            std::string( "##ViewportColors" );
        ImGui::ColorEdit4( label.c_str(), &viewportColors_[i].x );
    }
    ImGui::Separator();

    ImGui::Separator();
    ImGui::Text( "ImGui preset:" );
    UI::radioButton( "Dark", ( int* ) &preset_, int( ColorTheme::Preset::Dark ) );
    ImGui::SameLine();
    UI::radioButton( "Light", ( int* ) &preset_, int( ColorTheme::Preset::Light ) );

    ImGui::Separator();
    UI::checkbox( "Apply to new objects only", &applyToNewObjectsOnly_ );
    ImGui::SetNextItemWidth( 150.0f * menuScaling );
    UI::inputText( "Theme name", themeName_ );
    bool valid = !themeName_.empty() && !hasProhibitedChars( themeName_ );
    if ( UI::button( "Apply & Save", valid, Vector2f( -1, 0 ) ) )
    {
        std::error_code ec;
        auto saveDir = ColorTheme::getUserThemesDirectory() / ( asU8String( themeName_ ) + u8".json" );
        if ( std::filesystem::is_regular_file( saveDir, ec ) )
        {
            ImGui::OpenPopup( "File already exists" );
            incrementForceRedrawFrames();
        }
        else
        {
            auto error = save_();
            if ( !error.empty() )
                showError( error );
        }
    }
    if ( !valid )
    {
        UI::setTooltipIfHovered( themeName_.empty() ?
            "Cannot save theme with empty name" :
            "Please do not any of these symbols: \? * / \\ \" < >", menuScaling );
    }

    ModalDialog modalDialog( "File already exists", {
        .headline = "File already exists",
        .text = "Theme with name " + themeName_ + " already exists, override it?",
    } );
    if ( modalDialog.beginPopup( menuScaling ) )
    {
        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );

        const float p = ImGui::GetStyle().ItemSpacing.x;
        const Vector2f btnSize{ ( ImGui::GetContentRegionAvail().x - p  ) / 2.f, 0 };

        if ( UI::buttonCommonSize( "Save", btnSize, ImGuiKey_Enter ) )
        {
            auto error = save_();
            if ( error.empty() )
                ImGui::CloseCurrentPopup();
            else
                showError( error );
        }
        ImGui::SameLine( 0, p );
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::PopStyleVar();

        modalDialog.endPopup( menuScaling );
    }

    ImGui::PopItemWidth();
    ImGui::EndCustomStatePlugin();
}

std::string AddCustomThemePlugin::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( !ColorTheme::isInitialized() )
        return "Color theme is not initialized";
    return "";
}

bool AddCustomThemePlugin::onEnable_()
{
    updateThemeNames_();
    themeName_ = "CustomTheme1";
    return true;
}

bool AddCustomThemePlugin::onDisable_()
{
    applyToNewObjectsOnly_ = true;
    sceneColors_.clear();
    ribbonColors_.clear();
    return true;
}

void AddCustomThemePlugin::updateThemeNames_()
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
        for ( const auto& entry : Directory{ userThemesDir, ec } )
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

    auto itemId = RibbonSchemaHolder::schema().items.find( "Viewer settings" );
    if ( itemId != RibbonSchemaHolder::schema().items.end() )
    {
        if ( auto viewerSettingsPlugin = std::dynamic_pointer_cast< ViewerSettingsPlugin >( itemId->second.item ) )
        {
            if ( viewerSettingsPlugin->isActive() )
                viewerSettingsPlugin->updateThemes();
        }
    }
    update_();
}

Json::Value AddCustomThemePlugin::makeJson_()
{
    Json::Value root;

    for ( int i = 0; i < SceneColors::Count; ++i )
        serializeToJson( Color( sceneColors_[i] ), root[SceneColors::getName( SceneColors::Type( i ) )] );

    auto& viewportColors = root["Viewport Colors"];
    for ( int i = 0; i<int( ColorTheme::ViewportColorsType::Count ); ++i )
        serializeToJson( Color( viewportColors_[i] ), viewportColors[ColorTheme::getViewportColorTypeName( ColorTheme::ViewportColorsType( i ) )] );

    root["ImGuiPreset"] = ColorTheme::getPresetName( preset_ );

    auto& ribbonColors = root["Ribbon Colors"];
    for ( int i = 0; i<int( ColorTheme::RibbonColorsType::Count ); ++i )
        serializeToJson( Color( ribbonColors_[i] ), ribbonColors[ColorTheme::getRibbonColorTypeName( ColorTheme::RibbonColorsType( i ) )] );

    return root;
}

void AddCustomThemePlugin::update_()
{
    Json::Value backupTheme;
    ColorTheme::serializeCurrentToJson( backupTheme );

    if ( selectedUserPreset_ == 0 )
        ColorTheme::setupDefaultDark();
    else if ( selectedUserPreset_ == 1 )
        ColorTheme::setupDefaultLight();
    else
        ColorTheme::setupUserTheme( userThemesPresets_[selectedUserPreset_] );

    sceneColors_.resize( SceneColors::Count );
    for ( int i = 0; i < SceneColors::Count; ++i )
        sceneColors_[i] = Vector4f( SceneColors::get( SceneColors::Type( i ) ) );

    preset_ = ColorTheme::getPreset();

    viewportColors_.resize( int( ColorTheme::ViewportColorsType::Count ) );
    for ( int i = 0; i<int( ColorTheme::ViewportColorsType::Count ); ++i )
        viewportColors_[i] = Vector4f( ColorTheme::getViewportColor( ColorTheme::ViewportColorsType( i ) ) );

    ribbonColors_.resize( int( ColorTheme::RibbonColorsType::Count ) );
    for ( int i = 0; i<int( ColorTheme::RibbonColorsType::Count ); ++i )
        ribbonColors_[i] = Vector4f( ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType( i ) ) );

    ColorTheme::setupFromJson( backupTheme, ColorTheme::getThemeType() );
}

std::string AddCustomThemePlugin::save_()
{
    {
        std::error_code ec;
        auto saveDir = ColorTheme::getUserThemesDirectory() / ( asU8String( themeName_ ) + u8".json" );
        std::filesystem::create_directories( saveDir.parent_path(), ec );

        if ( !serializeJsonValue( makeJson_(), saveDir ) )
        {
            spdlog::error( "Color theme serialization failed: cannot write file {}", utf8string( saveDir ) );
            return "Cannot save theme with name: \"" + themeName_ + "\"";
        }
    }

    ColorTheme::setupUserTheme( themeName_ );
    ColorTheme::apply();
    if ( !applyToNewObjectsOnly_ )
    {
        auto visualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get() );
        for ( auto obj : visualObjs )
            obj->resetColors();
    }
    updateThemeNames_();
    return {};
}

MR_REGISTER_RIBBON_ITEM( AddCustomThemePlugin )

}
