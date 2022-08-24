#pragma once

#include "MRViewerFwd.h"
#include <string>
#include <functional>

namespace MR
{

class MRVIEWER_CLASS ViewerSetup
{
public:
    virtual ~ViewerSetup() = default;
    // Setups Viewer::LaunchParams, it should have arc and argv already set
    MRVIEWER_API virtual void setupLaunchParams( LaunchParams& launchParams ) const;
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

// Returns last registered ViewerSetup, this is called from main function before application is started
MRVIEWER_API std::unique_ptr<ViewerSetup> getRegisterViewerSetup();

using ViewerSetupConstructorLambda = std::function<std::unique_ptr<ViewerSetup>()>;

class MRVIEWER_CLASS RegisterViewerSetupConstructor
{
public:
    MRVIEWER_API RegisterViewerSetupConstructor( ViewerSetupConstructorLambda lambda );
};

// This macro register ViewerSetup class, for application, note that it overrides old registered ViewerSetup class
#define MR_REGISTER_VIEWER_SETUP( ViewerSetupType )\
static MR::RegisterViewerSetupConstructor __viewerSetupRegistrator##ViewerSetupType{[](){return std::make_unique<ViewerSetupType>();}};

// Register default ViewerSetup
MR_REGISTER_VIEWER_SETUP( ViewerSetup )

}
