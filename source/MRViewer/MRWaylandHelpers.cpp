#include "MRWaylandHelpers.h"
#ifdef MRVIEWER_EXPERIMENTAL_WAYLAND_SUPPORT
#include "MRGladGlfw.h"

#include "MRMesh/MRFinally.h"
#include "MRPch/MRSpdlog.h"

#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#include "xdg-foreign-unstable-v2-client-protocol.h"

#include <cstring>

namespace MR
{

WaylandExportedToplevel::WaylandExportedToplevel( GLFWwindow* window )
{
    // GLFW assigns the xdg_toplevel role only to a visible window, and exporting a surface without it is a fatal protocol error
    if ( !glfwGetWindowAttrib( window, GLFW_VISIBLE ) )
        return;

    auto* display = glfwGetWaylandDisplay();
    auto* queue = wl_display_create_queue( display );
    MR_FINALLY { wl_event_queue_destroy( queue ); };

    auto* displayWrapper = static_cast<wl_display*>( wl_proxy_create_wrapper( display ) );
    wl_proxy_set_queue( reinterpret_cast<wl_proxy*>( displayWrapper ), queue );
    auto* registry = wl_display_get_registry( displayWrapper );
    wl_proxy_wrapper_destroy( displayWrapper );
    MR_FINALLY { wl_registry_destroy( registry ); };

    zxdg_exporter_v2* exporter = nullptr;
    static constexpr wl_registry_listener cRegistryListener {
        .global = [] ( void* data, wl_registry* registry, uint32_t name, const char* interface, uint32_t )
        {
            if ( std::strcmp( interface, zxdg_exporter_v2_interface.name ) == 0 )
                *static_cast<zxdg_exporter_v2**>( data ) = static_cast<zxdg_exporter_v2*>( wl_registry_bind( registry, name, &zxdg_exporter_v2_interface, 1 ) );
        },
        .global_remove = [] ( void*, wl_registry*, uint32_t ) {},
    };
    wl_registry_add_listener( registry, &cRegistryListener, &exporter );
    wl_display_roundtrip_queue( display, queue );
    if ( !exporter )
    {
        spdlog::warn( "Wayland compositor does not support xdg-foreign-unstable-v2" );
        return;
    }
    MR_FINALLY { zxdg_exporter_v2_destroy( exporter ); };

    auto* exportedToplevel = zxdg_exporter_v2_export_toplevel( exporter, glfwGetWaylandWindow( window ) );
    static constexpr zxdg_exported_v2_listener cExportedListener {
        .handle = [] ( void* data, zxdg_exported_v2*, const char* exportedHandle )
        {
            *static_cast<std::string*>( data ) = exportedHandle;
        },
    };
    zxdg_exported_v2_add_listener( exportedToplevel, &cExportedListener, &handle );
    wl_display_roundtrip_queue( display, queue );
    // the private queue is destroyed on return
    wl_proxy_set_queue( reinterpret_cast<wl_proxy*>( exportedToplevel ), nullptr );
    exported = exportedToplevel;
}

WaylandExportedToplevel::~WaylandExportedToplevel()
{
    if ( exported )
        zxdg_exported_v2_destroy( static_cast<zxdg_exported_v2*>( exported ) );
}

} // namespace MR
#endif
