#include "MRColorTheme.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRRibbonButtonDrawer.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include <imgui.h>
#include <assert.h>
#include <fstream>

#ifdef __EMSCRIPTEN__

extern "C" 
{
// 0 - dark
// 1 - light
EMSCRIPTEN_KEEPALIVE int emsChangeColorTheme( int theme )
{
    using namespace MR;
    if ( theme == 0 )
        ColorTheme::setupDefaultDark();
    else
        ColorTheme::setupDefaultLight();
    ColorTheme::apply();
    return 1;
}

}
#endif

namespace MR
{

void ColorTheme::setupFromJson( const Json::Value& root )
{
    auto& instance = ColorTheme::instance_();

    bool success = true;

    if ( instance.sceneColors_.size() < SceneColors::Count )
        instance.sceneColors_.resize( SceneColors::Count );
    for ( int i = 0; i < SceneColors::Count; ++i )
    {
        auto name = SceneColors::getName( SceneColors::Type( i ) );
        if ( !root[name].isObject() )
        {
            success = false;
            break;
        }
        deserializeFromJson( root[name], instance.sceneColors_[i] );
    }

    if ( success )
    {
        if ( root["Ribbon Colors"].isObject() )
        {
            const auto& ribColors = root["Ribbon Colors"];
            for ( int i = 0; i < int( RibbonColorsType::Count ); ++i )
            {
                auto name =getRibbonColorTypeName( RibbonColorsType( i ) );
                if ( !ribColors[name].isObject() )
                {
                    success = false;
                    break;
                }
                deserializeFromJson( ribColors[name], instance.newUIColors_[i] );
            }
        }
        else
            success = false;
    }

    if ( success )
    {
        if ( root["Viewport Colors"].isObject() )
        {
            const auto& viewportColors = root["Viewport Colors"];
            for ( int i = 0; i < int( ViewportColorsType::Count ); ++i )
            {
                auto name = getViewportColorTypeName( ViewportColorsType( i ) );
                if ( !viewportColors[name].isObject() )
                {
                    success = false;
                    break;
                }
                deserializeFromJson( viewportColors[name], instance.viewportColors_[i] );
            }
        }
        else
            success = false;
    }

    if ( success )
    {
        if ( root["ImGuiPreset"].isString() )
            instance.themePreset_ = root["ImGuiPreset"].asString() == getPresetName( Preset::Light ) ? Preset::Light : Preset::Dark;
        else
            success = false;
    }

    if ( !success )
    {
        spdlog::error( "Color theme deserialization failed: invalid json schema." );
        instance.sceneColors_.clear();
    }
}

void ColorTheme::setupFromFile( const std::filesystem::path& path )
{
    auto res = deserializeJsonValue( path );
    if ( !res )
    {
        spdlog::error( "Color theme deserialization failed: {}", res.error() );
        return;
    }

    const auto& root = res.value();
    return setupFromJson( root );
}

void ColorTheme::serializeCurrentToFile( const std::filesystem::path& path )
{
    Json::Value root;
    serializeCurrentToJson( root );

    std::ofstream ofs( path );
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( !ofs || writer->write( root, &ofs ) != 0 )
    {
        spdlog::error( "Color theme serialization failed: cannot write file {}", utf8string( path ) );
    }

    ofs.close();
}

void ColorTheme::serializeCurrentToJson( Json::Value& root )
{
    assert( ImGui::GetCurrentContext() );
    auto& instance = ColorTheme::instance_();

    if ( instance.sceneColors_.size() < SceneColors::Count )
        instance.sceneColors_.resize( SceneColors::Count );

    for ( int i = 0; i < SceneColors::Count; ++i )
        instance.sceneColors_[i] = SceneColors::get( SceneColors::Type( i ) );

    const auto& vpParams = Viewer::instanceRef().viewport().getParameters();
    setViewportColor( vpParams.backgroundColor, ViewportColorsType::Background );
    setViewportColor( vpParams.borderColor, ViewportColorsType::Borders );

    for ( int i = 0; i < SceneColors::Count; ++i )
        serializeToJson( instance.sceneColors_[i], root[SceneColors::getName( SceneColors::Type( i ) )] );

    root["ImGuiPreset"] = getPresetName( instance.themePreset_ );

    auto& ribbonColors = root["Ribbon Colors"];
    for ( int i = 0; i<int( RibbonColorsType::Count ); ++i )
        serializeToJson( instance.newUIColors_[i], ribbonColors[getRibbonColorTypeName( RibbonColorsType( i ) )] );

    auto& vieportColors = root["Viewport Colors"];
    for ( int i = 0; i<int( ViewportColorsType::Count ); ++i )
        serializeToJson( instance.viewportColors_[i], vieportColors[getViewportColorTypeName( ViewportColorsType( i ) )] );
}

void ColorTheme::apply()
{
    assert( ColorTheme::isInitialized() );

    const auto& instance = ColorTheme::instance_();

    for ( int i = 0; i < SceneColors::Count; ++i )
        SceneColors::set( SceneColors::Type( i ), instance.sceneColors_[i] );

    RibbonButtonDrawer::InitGradientTexture();

    CommandLoop::appendCommand( [&] ()
    {
        resetImGuiStyle();

        auto& viewer = getViewerInstance();
        for ( auto& vp : viewer.viewport_list )
        {
            auto params = vp.getParameters();
            params.backgroundColor = getViewportColor( ViewportColorsType::Background );
            params.borderColor = getViewportColor( ViewportColorsType::Borders );
            vp.setParameters( params );
        }

        if ( viewer.globalBasisAxes )
            viewer.globalBasisAxes->setLabelsColor( SceneColors::get( SceneColors::Type::Labels ) );

        if ( viewer.basisAxes )
            viewer.basisAxes->setLabelsColor( SceneColors::get( SceneColors::Type::Labels ) );
    } );
}

bool ColorTheme::isInitialized()
{
    return !ColorTheme::instance_().sceneColors_.empty();
}

void ColorTheme::setRibbonColor( const Color& color, RibbonColorsType type )
{
    auto& instance = ColorTheme::instance_();
    instance.newUIColors_[int( type )] = color;
}

const Color& ColorTheme::getRibbonColor( RibbonColorsType type )
{
    const auto& instance = ColorTheme::instance_();
    return instance.newUIColors_[int( type )];
}

ColorTheme::Preset ColorTheme::getPreset()
{
    return instance_().themePreset_;
}

const char* ColorTheme::getPresetName( Preset type )
{
    constexpr std::array<const char*, size_t( Preset::Count )> presetNames =
    {
        "Dark",
        "Light"
    };
    return presetNames[int( type )];
}

ColorTheme::Type ColorTheme::getThemeType()
{
    return instance_().type_;
}

const std::string& ColorTheme::getThemeName()
{
    return instance_().themeName_;
}

void ColorTheme::setupByTypeName( Type type, const std::string& name )
{
    if ( type == Type::User )
        setupUserTheme( name );
    else if ( name == getPresetName( Preset::Light ) )
        setupDefaultLight();
    else
        setupDefaultDark();
}

void ColorTheme::setupDefaultDark()
{
    instance_().type_ = Type::Default;
    instance_().themeName_ = getPresetName( Preset::Dark );
    setupFromFile( MR::GetResourcesDirectory() / "MRDarkTheme.json" );
}

void ColorTheme::setupDefaultLight()
{
    instance_().type_ = Type::Default;
    instance_().themeName_ = getPresetName( Preset::Light );
    setupFromFile( MR::GetResourcesDirectory() / "MRLightTheme.json" );
}

void ColorTheme::setupUserTheme( const std::string& themeName )
{
    instance_().type_ = Type::User;
    instance_().themeName_ = themeName;
    setupFromFile( getUserThemesDirectory() / ( themeName + ".json" ) );
}

ColorTheme& ColorTheme::instance_()
{
    static ColorTheme instance;
    return instance;
}

const char* ColorTheme::getRibbonColorTypeName( RibbonColorsType type )
{
    constexpr std::array<const char*, size_t( RibbonColorsType::Count )> colorNames
    {
        "Background",
        "HeaderBackground",
        "HeaderSeparator",
        "TopPanelBackground",
        "QuickAccessBackground",
        "Borders",

        "TabHovered",
        "TabClicked",
        "TabActive",
        "TabActiveHovered",
        "TabActiveClicked",

        "TabText",
        "TabActiveText",

        "ToolbarHovered",
        "ToolbarClicked",

        "Text",
        "TextEnabled",
        "TextDisabled",
        "TextSelectedBg",

        "RibbonButtonHovered",
        "RibbonButtonClicked",
        "RibbonButtonActive",
        "RibbonButtonActiveHovered",
        "RibbonButtonActiveClicked",

        "ThirdpartyBackground",
        "FrameBackground",
        "CollapseHeaderBackground",
        "SelectedObjectText",
        "SelectedObjectFrame",
        "GradientStart",
        "GradientEnd"
    };
    return colorNames[int( type )];
}

void ColorTheme::setViewportColor( const Color& color, ViewportColorsType type )
{
    auto& instance = ColorTheme::instance_();
    instance.viewportColors_[int( type )] = color;
}

const Color& ColorTheme::getViewportColor( ViewportColorsType type )
{
    const auto& instance = ColorTheme::instance_();
    return instance.viewportColors_[int( type )];
}

const char* ColorTheme::getViewportColorTypeName( ViewportColorsType type )
{
    constexpr std::array<const char*, size_t( ViewportColorsType::Count )> colorNames
    {
        "Background",
        "Borders"
    };
    return colorNames[int( type )];
}

std::filesystem::path ColorTheme::getUserThemesDirectory()
{
    auto configDir = getUserConfigDir();
    configDir /= "UserThemes";
    return configDir;
}

void ColorTheme::resetImGuiStyle()
{
    const auto& instance = ColorTheme::instance_();

    auto& style = ImGui::GetStyle();
    style = ImGuiStyle();

    switch ( instance.themePreset_ )
    {
        case MR::ColorTheme::Preset::Light:
            ImGui::StyleColorsLight();
            break;
        case MR::ColorTheme::Preset::Dark:
        default:
            ImGui::StyleColorsDark();
            break;
    }

    Vector4f bg = Vector4f( getRibbonColor( RibbonColorsType::Background ) );
    Vector4f text = Vector4f( getRibbonColor( RibbonColorsType::Text ) );
    Vector4f border = Vector4f( getRibbonColor( RibbonColorsType::Borders ) );
    Vector4f frameBg = Vector4f( getRibbonColor( RibbonColorsType::FrameBackground ) );
    Vector4f headerBg = Vector4f( getRibbonColor( RibbonColorsType::CollapseHeaderBackground ) );
    Vector4f textSelBg = Vector4f( getRibbonColor( RibbonColorsType::TextSelectedBg ) );

    style.Colors[ImGuiCol_WindowBg] = ImVec4( bg.x, bg.y, bg.z, bg.w );
    style.Colors[ImGuiCol_Text] = ImVec4( text.x, text.y, text.z, text.w );
    style.Colors[ImGuiCol_Border] = ImVec4( border.x, border.y, border.z, border.w );
    style.Colors[ImGuiCol_FrameBg] = ImVec4( frameBg.x, frameBg.y, frameBg.z, frameBg.w );
    style.Colors[ImGuiCol_Header] = ImVec4( headerBg.x, headerBg.y, headerBg.z, headerBg.w );
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4( textSelBg.x, textSelBg.y, textSelBg.z, textSelBg.w );

    style.FrameRounding = 5.0f;
    style.FramePadding.y = 5.0f;
    style.ItemSpacing.y = 6.0f;

    style.FrameBorderSize = 1.0f;
    style.AntiAliasedLines = false;

    style.WindowBorderSize = 1.0f;
}

}