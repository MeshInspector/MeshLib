#include "MRAddCustomTheme.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRSceneColors.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRViewer/MRRibbonButtonDrawer.h"

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

    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth,.height = menuHeight, .menuScaling = menuScaling } ) )
        return;

    ImGui::PushItemWidth( 220.0f * menuScaling );
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
    RibbonButtonDrawer::GradientRadioButton( "Dark", ( int* ) &preset_, int( ColorTheme::Preset::Dark ) );
    ImGui::SameLine();
    RibbonButtonDrawer::GradientRadioButton( "Light", ( int* ) &preset_, int( ColorTheme::Preset::Light ) );

    ImGui::Separator();
    RibbonButtonDrawer::GradientCheckbox( "Apply to new objects only", &applyToNewObjectsOnly_ );
    ImGui::SetNextItemWidth( 150.0f * menuScaling );
    ImGui::InputText( "Theme name", themeName_ );
    if ( RibbonButtonDrawer::GradientButton( "Apply & Save", ImVec2( -1, 0 ) ) )
    {
        std::error_code ec;
        auto saveDir = ColorTheme::getUserThemesDirectory() / ( themeName_ + ".json" );
        if ( std::filesystem::is_regular_file( saveDir, ec ) )
        {
            ImGui::OpenPopup( "File already exists" );
            getViewerInstance().incrementForceRedrawFrames();
        }
        else
        {
            save_();
        }
    }

    if ( ImGui::BeginModalNoAnimation( "File already exists", nullptr,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize ) )
    {

        ImGui::Text( "Theme with name %s already exists, override it?", themeName_.c_str() );
        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( RibbonButtonDrawer::GradientButtonCommonSize( "Save", ImVec2( ( w - p ) / 2.f, 0 ), ImGuiKey_Enter ) )
        {
            save_();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine( 0, p );
        if ( RibbonButtonDrawer::GradientButtonCommonSize( "Cancel", ImVec2( ( w - p ) / 2.f, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
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
    update_();
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
}

void AddCustomThemePlugin::save_()
{
    {
        std::error_code ec;
        auto saveDir = ColorTheme::getUserThemesDirectory() / ( themeName_ + ".json" );
        std::filesystem::create_directories( saveDir.parent_path(), ec );

        auto json = makeJson_();
        std::ofstream ofs( saveDir );
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
        if ( !ofs || writer->write( json, &ofs ) != 0 )
        {
            spdlog::error( "Color theme serialization failed: cannot write file {}", utf8string( saveDir ) );
            return;
        }
    }

    ColorTheme::setupUserTheme( themeName_ );
    ColorTheme::apply();
    if ( !applyToNewObjectsOnly_ )
    {
        auto visualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get() );
        for ( auto obj : visualObjs )
        {
            obj->setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
            obj->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
            obj->setBackColor( SceneColors::get( SceneColors::BackFaces ) );
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
            obj->setLabelsColor( SceneColors::get( SceneColors::Labels ) );
MR_SUPPRESS_WARNING_POP
#ifndef __EMSCRIPTEN__
            if ( auto objVoxels = std::dynamic_pointer_cast< ObjectVoxels >( obj ) )
            {
                objVoxels->setFrontColor( SceneColors::get( SceneColors::SelectedObjectVoxels ), true );
                objVoxels->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectVoxels ), false );
            }
            else
#endif
            if ( auto objDM = std::dynamic_pointer_cast< ObjectDistanceMap >( obj ) )
            {
                objDM->setFrontColor( SceneColors::get( SceneColors::SelectedObjectDistanceMap ), true );
                objDM->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectDistanceMap ), false );
            }
            else if ( auto meshObj = std::dynamic_pointer_cast< ObjectMesh >( obj ) )
            {
                meshObj->setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
                meshObj->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
                meshObj->setSelectedFacesColor( SceneColors::get( SceneColors::SelectedFaces ) );
                meshObj->setSelectedEdgesColor( SceneColors::get( SceneColors::SelectedEdges ) );
                meshObj->setEdgesColor( SceneColors::get( SceneColors::Edges ) );
            }
            else if ( auto objPoints = std::dynamic_pointer_cast< ObjectPoints >( obj ) )
            {
                objPoints->setFrontColor( SceneColors::get( SceneColors::SelectedObjectPoints ), true );
                objPoints->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectPoints ), false );
            }
            else if ( auto objLines = std::dynamic_pointer_cast< ObjectLines >( obj ) )
            {
                objLines->setFrontColor( SceneColors::get( SceneColors::SelectedObjectLines ), true );
                objLines->setFrontColor( SceneColors::get( SceneColors::UnselectedObjectLines ), false );
            }
        }
    }
}

MR_REGISTER_RIBBON_ITEM( AddCustomThemePlugin )

}