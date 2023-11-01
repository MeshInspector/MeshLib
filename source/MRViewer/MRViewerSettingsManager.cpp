#include "MRViewerSettingsManager.h"
#include "MRMeshViewport.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRConfig.h"
#include "MRMeshViewer.h"
#include "MRColorTheme.h"
#include "MRRibbonMenu.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRMesh/MRSerializer.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRGLMacro.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRSpaceMouseHandlerHidapi.h"

namespace
{
const std::string cOrthogrphicParamKey = "orthographic";
const std::string cFlatShadingParamKey = "flatShading";
const std::string cGLPickRadiusParamKey = "glPickRadius";
const std::string cColorThemeParamKey = "colorTheme";
const std::string cSceneControlParamKey = "sceneControls";
const std::string cTopPanelPinnedKey = "topPanelPinned";
const std::string cQuickAccesListKey = "quickAccesList";
const std::string cMainWindowSize = "mainWindowSize";
const std::string cMainWindowPos = "mainWindowPos";
const std::string cMainWindowMaximized = "mainWindowMaximized";
const std::string cRibbonLeftWindowSize = "ribbonLeftWindowSize";
const std::string cShowSelectedObjects = "showSelectedObjects";
const std::string lastExtentionsParamKey = "lastExtentions";
const std::string cSpaceMouseSettings = "spaceMouseSettings";
const std::string cMSAA = "multisampleAntiAliasing";
const std::string cncMachineSettingsKey = "CNCMachineSettings";
const std::string cTouchpadSettings = "touchpadSettings";
}

namespace MR
{

ViewerSettingsManager::ViewerSettingsManager()
{
    lastExtentions_.resize( int( ObjType::Count ) );
}

int ViewerSettingsManager::loadInt( const std::string& name, int def )
{
    auto& cfg = Config::instance();
    if ( !cfg.hasJsonValue( name ) )
        return def;
    const auto& value = cfg.getJsonValue( name );
    if ( !value.isInt() )
        return def;
    return value.asInt();
}

void ViewerSettingsManager::saveInt( const std::string& name, int value )
{
    Json::Value val = value;
    Config::instance().setJsonValue( name, val );
}

void ViewerSettingsManager::loadSettings( Viewer& viewer )
{
    auto& viewport = viewer.viewport();
    auto params = viewport.getParameters();
    auto& cfg = Config::instance();
    params.orthographic = cfg.getBool( cOrthogrphicParamKey, params.orthographic );
    viewport.setParameters( params );

    viewer.glPickRadius = uint16_t( loadInt( cGLPickRadiusParamKey, viewer.glPickRadius ) );

    auto ribbonMenu = viewer.getMenuPluginAs<RibbonMenu>();
    if ( ribbonMenu )
        ribbonMenu->pinTopPanel( cfg.getBool( cTopPanelPinnedKey, true ) );

    if ( cfg.hasJsonValue( cSceneControlParamKey ) )
    {
        const auto& controls = cfg.getJsonValue( cSceneControlParamKey );
        for ( int i = 0; i < int( MouseMode::Count ); ++i )
        {
            MouseMode mode = MouseMode( i );
            if ( mode == MouseMode::None )
                continue;
            auto modeName = getMouseModeString( mode );
            if ( !controls[modeName].isInt() )
                continue;
            int key = controls[modeName].asInt();
            if ( key == -1 )
                continue;
            viewer.mouseController.setMouseControl( MouseController::keyToMouseAndMod( key ), mode );
        }
    }

    // SceneSettings
    SceneSettings::set( SceneSettings::Type::MeshFlatShading, cfg.getBool( cFlatShadingParamKey, SceneSettings::get( SceneSettings::Type::MeshFlatShading ) ) );
    if ( cfg.hasJsonValue( cncMachineSettingsKey ) )
    {
        CNCMachineSettings cncSettings;
        cncSettings.loadFromJson( cfg.getJsonValue(cncMachineSettingsKey) );
        SceneSettings::setCNCMachineSettings( cncSettings );
    }

    ColorTheme::Type colorThemeType = ColorTheme::Type::Default;
    std::string colorThemeName = ColorTheme::getPresetName( ColorTheme::Preset::Default ); // default
    if ( cfg.hasJsonValue( cColorThemeParamKey ) )
    {
        const auto& presetCfg = cfg.getJsonValue( cColorThemeParamKey );
        if ( presetCfg.isObject() )
        {

            if ( presetCfg["TypeId"].isInt() )
                colorThemeType = ColorTheme::Type( presetCfg["TypeId"].asInt() );
            if ( presetCfg["Name"].isString() )
                colorThemeName = presetCfg["Name"].asString();
        }
    }
#ifndef __EMSCRIPTEN__
    if ( cfg.hasVector2i( cMainWindowSize ) )
    {
        const auto size = cfg.getVector2i( cMainWindowSize, Vector2i( 1280, 800 ) );
        if ( size.x > 0 && size.y > 0 )
        {
            CommandLoop::appendCommand( [&viewer, size]
            {
                spdlog::info( "Resize window: {} {}", size.x, size.y );
                viewer.resize( size.x, size.y );
            } );
        }
    }
    if ( cfg.hasVector2i( cMainWindowPos ) )
    {
        auto pos = cfg.getVector2i( cMainWindowPos, Vector2i( 100, 100 ) );
        if ( pos.x > -32000 && pos.y > -32000 )
        {
            if ( pos.y <= 0 )
                pos.y = 40; // handle for one rare issue
            CommandLoop::appendCommand( [&viewer, pos]
            {
                if ( viewer.window )
                {
                    int count;
                    auto monitors = glfwGetMonitors( &count );
                    bool posIsOk = false;
                    for ( int i = 0; !posIsOk && i < count; ++i )
                    {
                        int xpos, ypos, width, height;
                        glfwGetMonitorWorkarea( monitors[i], &xpos, &ypos, &width, &height );
                        Box2i monBox = Box2i::fromMinAndSize( { xpos,ypos }, { width,height } );
                        posIsOk = monBox.contains( pos );
                    }
                    if ( posIsOk )
                    {
                        spdlog::info( "Set window pos: {} {}", pos.x, pos.y );
                        glfwSetWindowPos( viewer.window, pos.x, pos.y );
                    }
                }
            } );
        }
    }
    if ( cfg.hasBool( cMainWindowMaximized ) )
    {
        const bool maximized = cfg.getBool( cMainWindowMaximized );
        CommandLoop::appendCommand( [&viewer, maximized]
        {
            if ( !viewer.window )
                return;
            if ( maximized )
            {
                spdlog::info( "Maximize window." );
                glfwMaximizeWindow( viewer.window );
            }
            else
            {
                spdlog::info( "Restore window." );
                glfwRestoreWindow( viewer.window );
            }
        } );
    }
#endif
    if ( ribbonMenu )
    {
        if ( cfg.hasJsonValue( cQuickAccesListKey ) )
            ribbonMenu->readQuickAccessList( cfg.getJsonValue( cQuickAccesListKey ) );

        auto sceneSize = cfg.getVector2i( cRibbonLeftWindowSize, Vector2i{ 310, 0 } );
        // it is important to be called after `cMainWindowMaximized` block
        // as far as scene size is clamped by window size in each frame
        CommandLoop::appendCommand( [ribbonMenu, sceneSize]
        {
            spdlog::info( "Set menu plugin scene window size: {} {}", sceneSize.x, sceneSize.y );
            ribbonMenu->setSceneSize( sceneSize );
        } );

        if ( cfg.hasBool( cShowSelectedObjects ) )
            ribbonMenu->setShowNewSelectedObjects( cfg.getBool( cShowSelectedObjects ) );
    }

    ColorTheme::setupByTypeName( colorThemeType, colorThemeName );
    if ( !ColorTheme::isInitialized() )
    {
        spdlog::warn( "Color theme was not setup successfully, try setup default dark theme." );
        // most likely we loaded some bad user theme file
        // setup default in this case
        ColorTheme::setupByTypeName( ColorTheme::Type::Default, ColorTheme::getPresetName( ColorTheme::Preset::Default ) );
    }
    ColorTheme::apply();

    Json::Value lastExtentions = cfg.getJsonValue( lastExtentionsParamKey );
    if ( lastExtentions.isArray() )
    {
        const int end = std::min( (int)lastExtentions.size(), (int)lastExtentions_.size() );
        for ( int i = 0; i < end; ++i )
            lastExtentions_[i] = lastExtentions[i].asString();
    }

    if ( cfg.hasJsonValue( cSpaceMouseSettings ) )
    {
        const auto& paramsJson = cfg.getJsonValue( cSpaceMouseSettings );
        SpaceMouseController::Params spaceMouseParams;
        if ( paramsJson.isMember( "translateScale" ) )
            deserializeFromJson( paramsJson["translateScale"], spaceMouseParams.translateScale );
        if ( paramsJson.isMember( "rotateScale" ) )
            deserializeFromJson( paramsJson["rotateScale"], spaceMouseParams.rotateScale );
        viewer.spaceMouseController.setParams( spaceMouseParams );

#ifdef _WIN32
        if ( paramsJson.isMember( "activeMouseScrollZoom" ) && paramsJson["activeMouseScrollZoom"].isBool() )
        {
            if ( auto spaceMouseHandler =  viewer.getSpaceMouseHandler() )
            {
                auto hidapiHandler = std::dynamic_pointer_cast< SpaceMouseHandlerHidapi >( spaceMouseHandler );
                if ( hidapiHandler )
                {
                    const bool activeMouseScrollZoom = paramsJson["activeMouseScrollZoom"].asBool();
                    hidapiHandler->activateMouseScrollZoom( activeMouseScrollZoom );
                }
            }
        }
#endif
    }

    if ( cfg.hasJsonValue( cTouchpadSettings ) )
    {
        const auto& object = cfg.getJsonValue( cTouchpadSettings );
        TouchpadController::Parameters parameters;
        if ( object.isMember( "ignoreKineticMoves" ) && object["ignoreKineticMoves"].isBool() )
        {
            parameters.ignoreKineticMoves = object["ignoreKineticMoves"].asBool();
        }
        if ( object.isMember( "cancellable" ) && object["cancellable"].isBool() )
        {
            parameters.cancellable = object["cancellable"].asBool();
        }
        if ( object.isMember( "swipeMode" ) && object["swipeMode"].isInt() )
        {
            const auto swipeMode = object["swipeMode"].asInt();
            if ( swipeMode >= 0 && swipeMode < TouchpadController::Parameters::SwipeModeCount )
                parameters.swipeMode = (TouchpadController::Parameters::SwipeMode)swipeMode;
            else
                spdlog::warn( "Incorrect value for {}.swipeMode", cTouchpadSettings );
        }
        viewer.touchpadController.setParameters( parameters );
    }
}

void ViewerSettingsManager::saveSettings( const Viewer& viewer )
{
    const auto& viewport = viewer.viewport();
    const auto& params = viewport.getParameters();
    auto& cfg = Config::instance();
    cfg.setBool( cOrthogrphicParamKey, params.orthographic );

    saveInt( cGLPickRadiusParamKey, viewer.glPickRadius );

    auto ribbonMenu = viewer.getMenuPluginAs<RibbonMenu>();
    if ( ribbonMenu )
        cfg.setBool( cTopPanelPinnedKey, ribbonMenu->isTopPannelPinned() );

    Json::Value sceneControls;
    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        if ( mode == MouseMode::None )
            continue;
        auto control = viewer.mouseController.findControlByMode( mode );
        int key = control ? MouseController::mouseAndModToKey( *control ) : -1;
        sceneControls[getMouseModeString( mode )] = key;
    }
    cfg.setJsonValue( cSceneControlParamKey, sceneControls );

    // SceneSettings
    cfg.setBool( cFlatShadingParamKey, SceneSettings::get( SceneSettings::Type::MeshFlatShading ) );
    Json::Value cnfCNCSettings = SceneSettings::getCNCMachineSettings().saveToJson();
    cfg.setJsonValue( cncMachineSettingsKey, cnfCNCSettings );


    Json::Value colorThemePreset;
    colorThemePreset["TypeId"] = int( ColorTheme::getThemeType() );
    colorThemePreset["Name"] = ColorTheme::getThemeName();

    cfg.setJsonValue( cColorThemeParamKey, colorThemePreset );

    if ( ribbonMenu )
    {
        auto& quickAccessList = ribbonMenu->getToolbar().getItemsList();
        Json::Value qaList = Json::arrayValue;
        qaList.resize( int( quickAccessList.size() ) );
        for ( int i = 0; i < quickAccessList.size(); ++i )
            qaList[i]["Name"] = quickAccessList[i];
        cfg.setJsonValue( cQuickAccesListKey, qaList );

        cfg.setVector2i( cRibbonLeftWindowSize, ribbonMenu->getSceneSize() );
    }

    Json::Value exts = Json::arrayValue;
    for ( int i = 0; i < lastExtentions_.size(); ++i )
        exts[i] = lastExtentions_[i];
    cfg.setJsonValue( lastExtentionsParamKey, exts );

    // this is necessary for older versions of the software not to crash on reading these settings
    Json::Value xtext;
    xtext["Count"] = 0;
    xtext["Nums"] = Json::arrayValue;
    cfg.setJsonValue( "lastExtextentions", xtext );

    cfg.setVector2i( cMainWindowSize, viewer.windowSaveSize );
    if ( viewer.windowSaveSize.x > 0 && viewer.windowSaveSize.y > 0 )
        cfg.setVector2i( cMainWindowPos, viewer.windowSavePos );
    cfg.setBool( cMainWindowMaximized, viewer.windowMaximized );

    if ( ribbonMenu )
    {
        cfg.setBool( cShowSelectedObjects, ribbonMenu->getShowNewSelectedObjects() );
    }

    Json::Value spaceMouseParamsJson;
    SpaceMouseController::Params spaceMouseParams = viewer.spaceMouseController.getParams();
    serializeToJson( spaceMouseParams.translateScale, spaceMouseParamsJson["translateScale"] );
    serializeToJson( spaceMouseParams.rotateScale, spaceMouseParamsJson["rotateScale"] );
#ifdef _WIN32
    if ( auto spaceMouseHandler = viewer.getSpaceMouseHandler() )
    {
        auto hidapinHandler = std::dynamic_pointer_cast< SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapinHandler )
        {
            spaceMouseParamsJson["activeMouseScrollZoom"] = hidapinHandler->isMouseScrollZoomActive();
        }
    }
#endif
    cfg.setJsonValue( cSpaceMouseSettings, spaceMouseParamsJson );

    Json::Value touchpadParametersJson;
    const auto& touchpadParameters = viewer.touchpadController.getParameters();
    touchpadParametersJson["ignoreKineticMoves"] = touchpadParameters.ignoreKineticMoves;
    touchpadParametersJson["cancellable"] = touchpadParameters.cancellable;
    touchpadParametersJson["swipeMode"] = (int)touchpadParameters.swipeMode;
    cfg.setJsonValue( cTouchpadSettings, touchpadParametersJson );
}

const std::string & ViewerSettingsManager::getLastExtention( ObjType objType )
{
    int objTypeInt = int( objType );
    if ( objTypeInt < 0 || objTypeInt >= int( ObjType::Count ) )
    {
        assert( false );
        const static std::string empty;
        return empty;
    }
    return lastExtentions_[objTypeInt];
}

void ViewerSettingsManager::setLastExtention( ObjType objType, std::string ext )
{
    int objTypeInt = int( objType );
    if ( objTypeInt < 0 || objTypeInt >= int( ObjType::Count ) )
    {
        assert( false );
        return;
    }
    lastExtentions_[objTypeInt] = std::move( ext );
}

} //namespace MR
