#include "MRMenu.h"
#include "MRFileDialog.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRBitSet.h"
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRHistoryStore.h"
#include "ImGuiHelpers.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRToFromEigen.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimer.h"

#include "MRMesh/MRSceneSettings.h"
#include "MRCommandLoop.h"
#include "MRRibbonButtonDrawer.h"
#include "MRColorTheme.h"
#include "MRShortcutManager.h"
#include <GLFW/glfw3.h>

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif


namespace
{


// goes up and down on scene tree, selecting objects with different modifiers


}

namespace MR
{

void Menu::init( MR::Viewer *_viewer )
{
    ImGuiMenu::init( _viewer );

    callback_draw_viewer_menu = [&] ()
    {
        // Draw parent menu content
        draw_mr_menu();
    };

    // Draw additional windows
    callback_draw_custom_window = [&] ()
    {
        draw_scene_list();
        draw_helpers();
        draw_custom_plugins();
    };

    setupShortcuts_();
}









void Menu::setObjectTreeState( const Object* obj, bool open )
{
    if ( obj )
        sceneOpenCommands_[obj] = open;
}

void Menu::tryRenameSelectedObject()
{
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return;
    renameBuffer_ = selected[0]->name();
    showRenameModal_ = true;
}

void Menu::allowObjectsRemoval( bool allow )
{
    allowRemoval_ = allow;
}

void Menu::allowSceneReorder( bool allow )
{
    allowSceneReorder_ = allow;
}







void Menu::add_modifier( std::shared_ptr<MeshModifier> modifier )
{
    if ( modifier )
        modifiers_.push_back( modifier );
}

      



void Menu::setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec )
{
    frameTimeMillisecThreshold_ = maxGoodTimeMillisec;
}





void Menu::PluginsCache::validate( const std::vector<ViewerPlugin*>& viewerPlugins )
{
    // if same then cache is valid
    if ( viewerPlugins == allPlugins_ )
        return;

    allPlugins_ = viewerPlugins;

    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
        sortedCustomPlufins_[t] = {};
    for ( const auto& plugin : allPlugins_ )
    {
        StateBasePlugin * customPlugin = dynamic_cast< StateBasePlugin* >( plugin );
        if ( customPlugin )
            sortedCustomPlufins_[int( customPlugin->getTab() )].push_back( customPlugin );
    }
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        auto& tabPlugins = sortedCustomPlufins_[t];
        std::sort( tabPlugins.begin(), tabPlugins.end(), [] ( const auto& a, const auto& b )
        {
            return a->sortString() < b->sortString();
        } );
    }
}

StateBasePlugin* Menu::PluginsCache::findEnabled() const
{
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        const auto& tabPlugins = sortedCustomPlufins_[t];
        for ( auto plug : tabPlugins )
            if ( plug->isEnabled() )
                return plug;
    }
    return nullptr;
}

const std::vector<StateBasePlugin*>& Menu::PluginsCache::getTabPlugins( StatePluginTabs tab ) const
{
    return sortedCustomPlufins_[int( tab )];
}

}
