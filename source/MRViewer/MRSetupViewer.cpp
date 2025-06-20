#include "MRSetupViewer.h"
#include "MRRibbonMenu.h"
#include "MRViewer.h"
#include "MRViewerSettingsManager.h"
#include "MRMouseController.h"
#include "MRHistoryStore.h"
#include "MRGladGlfw.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
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

static void resetSettings( Viewer* viewer );

void ViewerSetup::setupConfiguration( Viewer* viewer ) const
{
    assert( viewer );

    viewer->enableGlobalHistory( true );

    viewer->resetSettingsFunction = [oldFunction = viewer->resetSettingsFunction] ( Viewer* viewer )
    {
        oldFunction( viewer );
        resetSettings( viewer );
    };
    viewer->resetSettingsFunction( viewer );
}

void resetSettings( Viewer * viewer )
{
    viewer->glPickRadius = 3;

    viewer->mouseController().setMouseControl( { MouseButton::Right,0 }, MouseMode::Translation );
    MouseController::MouseControlKey rotKey = { MouseButton::Middle,0 };
#ifdef __APPLE__
    rotKey = { MouseButton::Left,0 };
#endif
#ifdef __EMSCRIPTEN__
    const size_t memLimit = size_t( 1024 ) * 1024 * 1024;
    viewer->scrollForce = 0.7f;
    bool hasMouse = bool( EM_ASM_INT( return hasMouse() ) );
    bool isMac = bool( EM_ASM_INT( return is_mac() ) );
    if ( !hasMouse || isMac )
        rotKey = { MouseButton::Left,0 };
#else
    // History store can occupy from 2Gb to half of system physical memory
    const size_t memLimit = std::max( size_t( 2 ) * 1024 * 1024 * 1024, getSystemMemory().physicalTotal / 2 );
#endif
    viewer->mouseController().setMouseControl( rotKey, MouseMode::Rotation );
    rotKey.mod = GLFW_MOD_CONTROL;
    viewer->mouseController().setMouseControl( rotKey, MouseMode::Roll );
    spdlog::info( "History memory limit: {}", bytesString( memLimit ) );
    viewer->getGlobalHistoryStore()->setMemoryLimit( memLimit );
    viewer->setSortDroppedFiles( true );
}

void ViewerSetup::setupExtendedLibraries() const
{
#ifndef __EMSCRIPTEN__
    MR_TIMER;
    // get library names and their loading priority from *.ui.json files
    std::vector<std::pair<std::string, int>> lib2priority;
    std::error_code ec;
    for ( auto entry : Directory{ SystemPath::getResourcesDirectory(), ec } )
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
        std::filesystem::path pluginPath = SystemPath::getPluginsDirectory();
#if _WIN32
        pluginPath /= libName + ".dll" ;
#elif defined __APPLE__
        pluginPath /= "lib" + libName + ".dylib" ;
#else
        pluginPath /= "lib" + libName + ".so";
#endif
        if ( exists( pluginPath, ec ) )
        {
            spdlog::info( "Loading library {} with priority {}", utf8string( libName ), priority );
            bool success = true;
            LoadedModule lm{ libName };
#if _WIN32
            lm.module = LoadLibraryW( pluginPath.wstring().c_str() );
            if ( !lm.module )
            {
                success = false;
                spdlog::error( "Load library {} error: {}", utf8string( pluginPath ), GetLastError() );
                assert( false );
            }
#else
            lm.module = dlopen( utf8string( pluginPath ).c_str(), RTLD_NOW | RTLD_GLOBAL );
            if ( !lm.module )
            {
                success = false;
                spdlog::error( "Load library {} error: {}", utf8string( pluginPath ), dlerror() );
                assert( false );
            }
#endif
            if ( success )
            {
                spdlog::info( "Load library {} was successful", utf8string( libName ) );
                loadedModules_.push_back( lm );
            }
        }
    }
#endif // ifndef __EMSCRIPTEN__
}

void ViewerSetup::unloadExtendedLibraries() const
{
#ifndef __EMSCRIPTEN__
    MR_TIMER;
    // unload in reverse order
    while ( !loadedModules_.empty() )
    {
        spdlog::info( "Unloading library {}", utf8string( loadedModules_.back().filename ) );
#if _WIN32
        FreeLibrary( loadedModules_.back().module );
#else
        dlclose( loadedModules_.back().module );
#endif //_WIN32
        spdlog::info( "Unload finished {}", utf8string( loadedModules_.back().filename ) );
        loadedModules_.pop_back();
    }
#endif // ifndef __EMSCRIPTEN__
}

} //namespace MR
