#include "MRSetupViewer.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"
#include "MRRibbonMenu.h"
#include "MRViewer.h"
#include "MRViewerSettingsManager.h"
#include "MRMesh/MRHistoryStore.h"
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

void ViewerSetup::setupConfiguration( Viewer* viewer ) const
{
    assert( viewer );

    viewer->defaultLabelsBasisAxes = true;
    viewer->enableGlobalHistory( true );

    viewer->mouseController.setMouseControl( { MouseButton::Right,0 }, MouseMode::Translation );

#ifndef __EMSCRIPTEN__
    viewer->mouseController.setMouseControl( { MouseButton::Middle,0 }, MouseMode::Rotation );
    // 2 GB for desktop version
    viewer->getGlobalHistoryStore()->setMemoryLimit( size_t( 2 ) * 1024 * 1024 * 1024 );
#else
    bool hasMouse = bool( EM_ASM_INT( return hasMouse() ) );
    if ( hasMouse )
        viewer->mouseController.setMouseControl( { MouseButton::Middle,0 }, MouseMode::Rotation );
    else
        viewer->mouseController.setMouseControl( { MouseButton::Left,0 }, MouseMode::Rotation );
    // 1 GB for WASM version
    viewer->getGlobalHistoryStore()->setMemoryLimit( size_t( 1024 ) * 1024 * 1024 );
    viewer->scrollForce = 0.7f;
#endif
}

void ViewerSetup::setupExtendedLibraries() const
{
    const auto pluginLibraryList = getPluginLibraryList();
    if ( !pluginLibraryList )
        return;
    for (const auto& pluginLib : *pluginLibraryList)
    {
#if _WIN32
        auto result = LoadLibraryW( pluginLib.wstring().c_str() );
        if ( !result )
        {
            auto error = GetLastError();
            spdlog::error( "Load library {} error: {}", utf8string( pluginLib ), error );
            assert( false );
        }
#else
        void* result = dlopen( utf8string( pluginLib ).c_str(), RTLD_LAZY );
        if ( !result )
        {
            auto error = dlerror();
            spdlog::error( "Load library {} error: {}", utf8string( pluginLib ), error );
            assert( false );
        }
#endif
    }
}

}
