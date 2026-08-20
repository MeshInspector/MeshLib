#pragma once

#include "MRCommonPlugins/exports.h"

namespace MR
{

/// binds default keyboard shortcuts to the ribbon items registered by this library;
/// shall be called by an application after the menu initialization, since MRViewer knows nothing about these items
MRCOMMONPLUGINS_API void setupCommonPluginsShortcuts();

} //namespace MR
