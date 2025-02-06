#pragma once

#include "MRViewerFwd.h"
#include <string>
#include <filesystem>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
   // For `HMODULE`. Even though some sources say it's `void *`, on MSVC it's apparently not one (but points to a special struct).
   // So we'd either need to cast back and forth to `void *`, or include the proper definition for it.
#  include <windows.h>
#endif

namespace MR
{

class MRVIEWER_CLASS ViewerSetup
{
public:
    /// explicitly define ctors to avoid warning C5267: definition of implicit copy constructor is deprecated because it has a user-provided destructor
    ViewerSetup() = default;
    ViewerSetup( const ViewerSetup & ) = default;
    ViewerSetup( ViewerSetup && ) = default;

    virtual ~ViewerSetup() = default;

    /// Setups Menu Save and Open plugins
    MRVIEWER_API virtual void setupBasePlugins( Viewer* /*viewer*/ ) const;

    /// Setups modifiers to Menu plugin if it is present in viewer
    virtual void setupCommonModifiers( Viewer* /*viewer*/ ) const {}

    /// Setups custom plugins to viewer
    virtual void setupCommonPlugins( Viewer* /*viewer*/ ) const {}

    /// Sets custom viewer settings manager to viewer
    MRVIEWER_API virtual void setupSettingsManager( Viewer* viewer, const std::string& appName ) const;

    /// Sets start configuration for viewer
    /// use to override default configuration of viewer
    MRVIEWER_API virtual void setupConfiguration( Viewer* viewer ) const;

    /// use to load additional libraries with plugins.
    /// all libraries are taken from GetResourcesDirectory() *.ui.json files
    MRVIEWER_API virtual void setupExtendedLibraries() const;

    /// free all libraries loaded in setupExtendedLibraries()
    MRVIEWER_API virtual void unloadExtendedLibraries() const;

private:
#ifndef __EMSCRIPTEN__
    struct LoadedModule
    {
        std::filesystem::path filename;
#if _WIN32
        HMODULE module = nullptr;
#else
        void * module = nullptr;
#endif
    };
    mutable std::vector<LoadedModule> loadedModules_;
#endif // ifndef __EMSCRIPTEN__
};

} // namespace MR
