#pragma once

#include "exports.h"
#include "MRRibbonMenuUIConfig.h"
#include "MRMesh/MRColor.h"
#include "MRPch/MRJson.h"
#include <optional>

namespace MR
{

struct RibbonConfig
{
    std::optional<RibbonMenuUIConfig> menuUIConfig;
    std::optional<Color> monochromeRibbonIcons; // if present all ribbon items will be drawn with this monochrome color (if available)
    std::optional<Json::Value> colorTheme; // if present will be applied
    std::optional<Json::Value> ribbonStructure; // analogue for ui.json file, if present will replace current layout
    std::optional<Json::Value> ribbonItemsOverrides; // analogue for items.json file, if present will override icon, caption, tooltip, droplist and helplink
};

// parse given json and setup `RibbonConfig` from it
MRVIEWER_API RibbonConfig createRibbonConfigFromJson( const Json::Value& root );

// apply given config to the application
MRVIEWER_API void applyRibbonConfig( const RibbonConfig& config );

} //namespace MR
