#pragma once
#ifdef MRVIEWER_EXPERIMENTAL_WAYLAND_SUPPORT

#include <string>

struct GLFWwindow;

namespace MR
{

/// Window exported via the xdg-foreign-unstable-v2 protocol; the export is revoked on destruction.
struct WaylandExportedToplevel
{
    /// Exports the window; leaves \ref exported null and \ref handle empty if the window is hidden
    /// or the compositor does not support the protocol.
    explicit WaylandExportedToplevel( GLFWwindow* window );
    ~WaylandExportedToplevel();

    WaylandExportedToplevel( const WaylandExportedToplevel& ) = delete;
    WaylandExportedToplevel& operator=( const WaylandExportedToplevel& ) = delete;

    /// opaque pointer to the zxdg_exported_v2 object
    void* exported = nullptr;
    /// handle of the exported window, valid while this object exists
    std::string handle;
};

} // namespace MR
#endif
