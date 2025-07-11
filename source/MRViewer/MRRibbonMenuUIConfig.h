#pragma once

namespace MR
{

enum class RibbonTopPanelLayoutMode
{
    None, ///< no top panel at all (toolbar is also forced hidden in this mode)
    RibbonNoTabs, ///< show only icons from the first tab, without tabs panel
    RibbonWithTabs ///< both ribbon toolbar and tabs
};

struct RibbonMenuUIConfig
{
    RibbonTopPanelLayoutMode topLayout{ RibbonTopPanelLayoutMode::RibbonWithTabs }; ///< how to show top panel
    bool centerRibbonItems{ false }; ///< if true - items on selected tab will be centered to have equal spacing from left and right (ignored top panel is hidden)

    bool drawScenePanel{ true }; ///< if false - panel with scene tree, information and transform will be hidden
    bool drawToolbar{ true }; ///< if false - toolbar will be hidden (ignored if top panel is hidden)
    bool drawViewportTags{ true }; ///< if false - window with viewport label and id will be hidden
    bool drawNotifications{ true }; ///< if false - no notifications are drawn on screen

    bool operator==( const RibbonMenuUIConfig& ) const = default;
};

} //namespace MR
