#include "MRSetupViewer.h"
#include "MRRibbonMenu.h"
#include "MRViewer.h"
#include "MRViewerSettingsManager.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRHistoryStore.h"
#include "MRMesh/MRDirectory.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRGladGlfw.h"
#if _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace MR
{

void ViewerSetup::setupBasePlugins( Viewer* viewer ) const
{
    assert( viewer );
    auto menu = std::make_shared<RibbonMenu>();
    viewer->setMenuPlugin( menu );
}

void ViewerSetup::setupSettingsManager( Viewer* viewer, const std::string& appName ) const
{
    assert( viewer );

    auto& cfg = MR::Config::instance();

    cfg.reset( appName );
    std::unique_ptr<ViewerSettingsManager> mng = std::make_unique<ViewerSettingsManager>();
    viewer->setViewportSettingsManager( std::move( mng ) );
}

void ViewerSetup::setupConfiguration( Viewer* viewer ) const
{
    assert( viewer );

    viewer->glPickRadius = 3;
    viewer->defaultLabelsBasisAxes = true;
    viewer->enableGlobalHistory( true );

    viewer->mouseController.setMouseControl( { MouseButton::Right,0 }, MouseMode::Translation );
    MouseController::MouseControlKey rotKey = { MouseButton::Middle,0 };
    size_t memLimit = size_t( 2 ) * 1024 * 1024 * 1024;
#ifdef __APPLE__
    rotKey = { MouseButton::Left,0 };
#endif
#ifdef __EMSCRIPTEN__
    memLimit = size_t( 1024 ) * 1024 * 1024;
    viewer->scrollForce = 0.7f;
    bool hasMouse = bool( EM_ASM_INT( return hasMouse() ) );
    bool isMac = bool( EM_ASM_INT( return is_mac() ) );
    if ( !hasMouse || isMac )
        rotKey = { MouseButton::Left,0 };
#endif
    viewer->mouseController.setMouseControl( rotKey, MouseMode::Rotation );
    rotKey.mod = GLFW_MOD_CONTROL;
    viewer->mouseController.setMouseControl( rotKey, MouseMode::Roll );
    viewer->getGlobalHistoryStore()->setMemoryLimit( memLimit );
}

void ViewerSetup::setupExtendedLibraries() const
{
#ifndef __EMSCRIPTEN__
    // get library names and their loading priority from *.ui.json files
    std::vector<std::pair<std::string, int>> lib2priority;
    std::error_code ec;
    for ( auto entry : Directory{ GetResourcesDirectory(), ec } )
    {
        if ( entry.path().u8string().ends_with( asU8String( ".ui.json" ) ) )
        {
            auto fileJson = deserializeJsonValue( entry.path() );
            if ( !fileJson  )
            {
                spdlog::error( "JSON ({}) deserialize error: {}", utf8string( entry.path().filename() ), fileJson.error() );
                assert( false );
                continue;
            }
            if ( !fileJson.value()["LibName"].isString() || !fileJson.value()["Order"].isInt())
            {
                spdlog::info( "JSON ({}) format error. Please, check the values of 'LibName' and/or 'Order' fields.", utf8string( entry.path().filename() ), fileJson.error() );
                assert( false );
                continue;
            }
            lib2priority.emplace_back( fileJson.value()["LibName"].asString(), fileJson.value()["Order"].asInt() );
        }
    }
    // sort by ascending priority
    std::sort( lib2priority.begin(), lib2priority.end(), []( auto& lhv, auto& rhv) { return lhv.second < rhv.second; } );

    for (const auto& [libName, priority] : lib2priority) {
        std::filesystem::path pluginPath = GetLibsDirectory();
#if _WIN32
        pluginPath /= libName + ".dll" ;
#elif defined __APPLE__
        pluginPath /= "lib" + libName + ".dylib" ;
#else
        pluginPath /= "lib" + libName + ".so";
#endif
        if ( exists(pluginPath) )
        {
            spdlog::info( "Loading library {} with priority {}", utf8string( libName ), priority );
#if _WIN32
            auto result = LoadLibraryW( pluginPath.wstring().c_str() );
            if ( !result )
            {
                spdlog::error( "Load library {} error: {}", utf8string( pluginPath ), GetLastError() );
                assert( false );
            }
#else
            auto result = dlopen( utf8string( pluginPath ).c_str(), RTLD_LAZY );
            if ( !result )
            {
                spdlog::error( "Load library {} error: {}", utf8string( pluginPath ), dlerror() );
                assert( false );
            }
#endif
        }
    }
#endif // ifndef __EMSCRIPTEN__
}
}

