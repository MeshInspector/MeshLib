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

}

}
