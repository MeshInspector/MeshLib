#include "MRCommonPluginsShortcuts.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRShortcutManager.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

void setupCommonPluginsShortcuts()
{
    auto menu = RibbonMenu::instance();
    if ( !menu )
    {
        spdlog::warn( "No RibbonMenu to setup MRCommonPlugins shortcuts in" );
        return;
    }

    menu->addRibbonItemShortcut( "Fit data", { GLFW_KEY_F, getGlfwModPrimaryCtrl() | GLFW_MOD_ALT }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Top View", { GLFW_KEY_KP_7, 0 }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Front View", { GLFW_KEY_KP_1, 0 }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Right View", { GLFW_KEY_KP_3, 0 }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Invert View", { GLFW_KEY_KP_9, 0 }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Bottom View", { GLFW_KEY_KP_7, getGlfwModPrimaryCtrl() }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Back View", { GLFW_KEY_KP_1, getGlfwModPrimaryCtrl() }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Left View", { GLFW_KEY_KP_3, getGlfwModPrimaryCtrl() }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Show_Hide Global Basis", { GLFW_KEY_G, getGlfwModPrimaryCtrl() }, ShortcutCategory::View );
    menu->addRibbonItemShortcut( "Select objects", { GLFW_KEY_Q, getGlfwModPrimaryCtrl() }, ShortcutCategory::Objects );
    menu->addRibbonItemShortcut( "Open files", { GLFW_KEY_O, getGlfwModPrimaryCtrl() }, ShortcutCategory::Scene );
    menu->addRibbonItemShortcut( "Save Scene", { GLFW_KEY_S, getGlfwModPrimaryCtrl() }, ShortcutCategory::Scene );
    menu->addRibbonItemShortcut( "Save Scene As", { GLFW_KEY_S, getGlfwModPrimaryCtrl() | GLFW_MOD_SHIFT }, ShortcutCategory::Scene );
    menu->addRibbonItemShortcut( "New", { GLFW_KEY_N, getGlfwModPrimaryCtrl() }, ShortcutCategory::Scene );
    menu->addRibbonItemShortcut( "Ribbon Scene Rename", { GLFW_KEY_F2, 0 }, ShortcutCategory::Objects );
    menu->addRibbonItemShortcut( "Ribbon Scene Remove selected objects", { GLFW_KEY_R, GLFW_MOD_SHIFT }, ShortcutCategory::Objects );
    menu->addRibbonItemShortcut( "Viewer settings", { GLFW_KEY_COMMA, getGlfwModPrimaryCtrl() }, ShortcutCategory::Info );
}

} //namespace MR
