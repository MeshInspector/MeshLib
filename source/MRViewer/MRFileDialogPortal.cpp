#include "MRFileDialogPortal.h"
#ifndef MRVIEWER_NO_XDG_DESKTOP_PORTAL
#include "MRViewer.h"

#include <dbus/dbus.h>

#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>

#include "MRMesh/MRStringConvert.h"

namespace
{

std::string getWindowId( GLFWwindow* window )
{
    // https://flatpak.github.io/xdg-desktop-portal/docs/window-identifiers.html
    switch ( glfwGetPlatform() )
    {
    case GLFW_PLATFORM_X11:
        return fmt::format( "x11:{:x}", glfwGetX11Window( window ) );
    // TODO: Wayland support
    default:
        return "";
    }
}

void dbusAppend( DBusMessageIter& iter, std::byte value )
{
    dbus_message_iter_append_basic( &iter, DBUS_TYPE_BYTE, &value );
}

void dbusAppend( DBusMessageIter& iter, bool value )
{
    int intValue = value;
    dbus_message_iter_append_basic( &iter, DBUS_TYPE_BOOLEAN, &intValue );
}

void dbusAppend( DBusMessageIter& iter, uint32_t value )
{
    dbus_message_iter_append_basic( &iter, DBUS_TYPE_UINT32, &value );
}

void dbusAppend( DBusMessageIter& iter, const char* str )
{
    dbus_message_iter_append_basic( &iter, DBUS_TYPE_STRING, &str );
}

# if 0
const char* signatureFor( std::byte )
{
    return DBUS_TYPE_BYTE_AS_STRING;
}
# endif

const char* signatureFor( bool )
{
    return DBUS_TYPE_BOOLEAN_AS_STRING;
}

const char* signatureFor( const char* )
{
    return DBUS_TYPE_STRING_AS_STRING;
}

struct ContainerHandle
{
    DBusMessageIter iter;
    DBusMessageIter* parent;

    ContainerHandle( DBusMessageIter& parent, int type, const char* signature )
        : parent( &parent )
    {
        dbus_message_iter_open_container( &parent, type, signature, &iter );
    }

    operator DBusMessageIter& ()
    {
        return iter;
    }

    ~ContainerHandle()
    {
        dbus_message_iter_close_container( parent, &iter );
    }
};

void dbusAppendArray( DBusMessageIter& iter, const char* signature, std::function<void ( DBusMessageIter& )> func )
{
    ContainerHandle arr( iter, DBUS_TYPE_ARRAY, signature );
    func( arr );
}

void dbusAppend( DBusMessageIter& iter, const char* key, const char* signature, std::function<void ( DBusMessageIter& )> func )
{
    ContainerHandle dictEntry( iter, DBUS_TYPE_DICT_ENTRY, nullptr );
    dbusAppend( dictEntry, key );

    ContainerHandle var( dictEntry, DBUS_TYPE_VARIANT, signature );
    func( var );
}

void dbusAppend( DBusMessageIter& iter, const char* key, auto&& value )
{
    dbusAppend( iter, key, signatureFor( value ), [&] ( auto& var )
    {
        dbusAppend( var, value );
    } );
}

void dbusAppendArray( DBusMessageIter& iter, const char* key, const char* signature, std::function<void ( DBusMessageIter& )> func )
{
    const auto arraySignature = std::string{ DBUS_TYPE_ARRAY_AS_STRING } + signature;
    dbusAppend( iter, key, arraySignature.c_str(), [&] ( auto& var )
    {
        dbusAppendArray( var, signature, func );
    } );
}

struct FileChooserFilter
{
    std::string name;
    std::string glob;
    std::string mime;

    enum Kind : uint32_t
    {
        GlobKind = 0,
        MimeKind = 1,
    };
};

struct FileChooserOptions
{
    std::string handleToken;
    std::string acceptLabel;
    bool modal = true;
    bool multiple = false;
    bool directory = false;
    std::vector<FileChooserFilter> filters;
    std::optional<FileChooserFilter> currentFilter;
    // TODO: choices
    std::string currentName;
    std::filesystem::path currentFolder;
    std::filesystem::path currentFile;
};

void dbusAppend( DBusMessageIter& iter, const FileChooserFilter& filter )
{
    ContainerHandle cont( iter, DBUS_TYPE_STRUCT, nullptr );
    dbusAppend( cont, filter.name.c_str() );
    ContainerHandle arr( cont, DBUS_TYPE_ARRAY, "(us)" );
    if ( !filter.glob.empty() )
    {
        ContainerHandle glob( arr, DBUS_TYPE_STRUCT, nullptr );
        dbusAppend( glob, FileChooserFilter::GlobKind );
        dbusAppend( glob, filter.glob.c_str() );
    }
    if ( !filter.mime.empty() )
    {
        ContainerHandle mime( arr, DBUS_TYPE_STRUCT, nullptr );
        dbusAppend( mime, FileChooserFilter::MimeKind );
        dbusAppend( mime, filter.mime.c_str() );
    }
}

const char* signatureFor( const FileChooserFilter& )
{
    return "(sa(us))";
}

std::vector<std::byte> toByteArray( const std::filesystem::path& path )
{
    const auto str = MR::utf8string( path );
    std::vector<std::byte> results;
    results.reserve( str.size() + 1 );
    for ( auto ch : str )
        results.emplace_back( (std::byte)ch );
    results.emplace_back( (std::byte)'\0' );
    return results;
}

void initArgs( DBusMessage* msg, const char* parentWindow, const char* title, const FileChooserOptions& options )
{
    DBusMessageIter args;
    dbus_message_iter_init_append( msg, &args );

    dbusAppend( args, parentWindow );
    dbusAppend( args, title );

    dbusAppendArray( args, "{sv}", [&] ( auto& dict )
    {
        if ( !options.handleToken.empty() )
            dbusAppend( dict, "handle_token", options.handleToken.c_str() );
        if ( !options.acceptLabel.empty() )
            dbusAppend( dict, "accept_label", options.acceptLabel.c_str() );
        if ( !options.modal )
            dbusAppend( dict, "modal", options.modal );
        if ( options.multiple )
            dbusAppend( dict, "multiple", options.multiple );
        if ( options.directory )
            dbusAppend( dict, "directory", options.directory );
        if ( !options.filters.empty() )
        {
            dbusAppendArray( dict, "filters", "(sa(us))", [&] ( auto& arr )
            {
                for ( const auto& filter : options.filters )
                    dbusAppend( arr, filter );
            } );
        }
        if ( options.currentFilter )
            dbusAppend( dict, "current_filter", *options.currentFilter );
        if ( !options.currentName.empty() )
            dbusAppend( dict, "current_name", options.currentName.c_str() );
        if ( !options.currentFolder.empty() )
        {
            dbusAppendArray( dict, "current_folder", DBUS_TYPE_BYTE_AS_STRING, [&] ( auto& arr )
            {
                for ( auto ch : toByteArray( options.currentFolder ) )
                    dbusAppend( arr, ch );
            } );
        }
    } );
}

} // namespace

namespace MR::detail
{

std::vector<std::filesystem::path> runPortalFileDialog( const MR::FileDialog::Parameters& params )
{
    DBusError err;
    dbus_error_init( &err );

    auto* conn = dbus_bus_get( DBUS_BUS_SESSION, &err );
    if ( dbus_error_is_set( &err ) )
    {
        spdlog::warn( "Failed to connect to DBus: {}", err.message );
        return {};
    }

    // https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.FileChooser.html
    constexpr auto* cSessionBus = "org.freedesktop.portal.Desktop";
    constexpr auto* cObjectPath = "/org/freedesktop/portal/desktop";
    constexpr auto* cInterface = "org.freedesktop.portal.FileChooser";
    const auto* method = "OpenFile";
    if ( params.saveDialog )
        method = params.folderDialog ? "SaveFiles" : "SaveFile";
    auto* msg = dbus_message_new_method_call( cSessionBus, cObjectPath, cInterface, method );

    const auto parentWindowId = getWindowId( getViewerInstance().window );

    const auto* title = params.multiselect ? "Open Files" : "Open File";
    if ( params.folderDialog )
        title = params.multiselect ? "Open Folders" : "Open Folder";
    else if ( params.saveDialog )
        title = params.multiselect ? "Save Files" : "Save File";

    FileChooserOptions options;
    options.currentFolder = params.baseFolder;
    for ( const auto& filter : params.filters )
    {
        // TODO: handle multiple extensions
        // TODO: handle case-sensitivity
        options.filters.push_back( {
            .name = filter.name,
            .glob = filter.extensions,
        } );
    }
    if ( !params.saveDialog )
    {
        options.multiple = params.multiselect;
        options.directory = params.folderDialog;
    }
    else
    {
        options.currentName = params.fileName;
    }

    initArgs( msg, parentWindowId.c_str(), title, options );

    auto* reply = dbus_connection_send_with_reply_and_block( conn, msg, DBUS_TIMEOUT_INFINITE, &err );
    dbus_connection_flush( conn );
    dbus_message_unref( msg );
    if ( dbus_error_is_set( &err ) )
    {
        spdlog::warn( "Failed to send DBus message: {}", err.message );
        return {};
    }

    std::vector<std::filesystem::path> results;
    DBusMessageIter replyIter;
    if ( dbus_message_iter_init( reply, &replyIter ) )
    {
        //
    }

    return results;
}

} // namespace MR::detail
#endif
