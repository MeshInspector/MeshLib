#pragma once

#include "MRViewerFwd.h"
#include <string>

namespace MR
{

class MRVIEWER_CLASS ViewerSetup
{
public:
    virtual ~ViewerSetup() = default;
    // Setups Menu Save and Open plugins
    MRVIEWER_API virtual void setupBasePlugins( Viewer* /*viewer*/ ) const;
    // Setups modifiers to Menu plugin if it is present in viewer
    virtual void setupCommonModifiers( Viewer* /*viewer*/ ) const {}
    // Setups custom plugins to viewer
    virtual void setupCommonPlugins( Viewer* /*viewer*/ ) const {}
    // Sets custom viewer settings manager to viewer
    MRVIEWER_API virtual void setupSettingsManager( Viewer* viewer, const std::string& appName ) const;
    // Sets start configuration for viewer
    // use to override default configuration of viewer
    MRVIEWER_API virtual void setupConfiguration( Viewer* viewer ) const;
    // use to load additional libraries with plugins. See pluginLibraryList.json
    MRVIEWER_API virtual void setupExtendedLibraries() const;
};
}
