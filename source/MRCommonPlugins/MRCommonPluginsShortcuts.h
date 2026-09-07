#pragma once

#include "MRCommonPlugins/exports.h"

namespace MR
{

/// does nothing: the items bind their default shortcuts from items.json themselves (see MenuItemShortcut)
[[deprecated( "the items bind their default shortcuts from items.json themselves, remove the call" )]]
MRCOMMONPLUGINS_API void setupCommonPluginsShortcuts();

} //namespace MR
