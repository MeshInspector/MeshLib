#include "MRFileDialogPortal.h"
#if !defined( _WIN32 ) && !defined( MRVIEWER_NO_XDG_DESKTOP_PORTAL )
#include "MRGladGlfw.h"
#include "MRViewer.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"

#include <dbus/dbus.h>

#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#undef Success // :(

namespace
{

DBusConnection* getConnection()
{
    static DBusConnection* sConn = []
    {
        DBusError err;
        dbus_error_init( &err );
        MR_FINALLY { dbus_error_free( &err ); };

        auto* conn = dbus_bus_get( DBUS_BUS_SESSION, &err );
        if ( dbus_error_is_set( &err ) )
            spdlog::warn( "Failed to connect to DBus: {}", err.message );

        return conn;
    } ();
    return sConn;
}

std::string getWindowId( GLFWwindow* window )
{
    // https://flatpak.github.io/xdg-desktop-portal/docs/window-identifiers.html
#if GLFW_VERSION_MAJOR > 3 || ( GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4 )
    switch ( glfwGetPlatform() )
    {
    case GLFW_PLATFORM_X11:
        return fmt::format( "x11:{:08x}", glfwGetX11Window( window ) );
    // TODO: Wayland support
    default:
        return "";
    }
#else
    return fmt::format( "x11:{:08x}", glfwGetX11Window( window ) );
#endif
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
    std::vector<std::string> globs;
    std::vector<std::string> mimes;

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
    // choices are not used for now
    std::string currentName;
    std::filesystem::path currentFolder;
    std::filesystem::path currentFile;
};

void dbusAppend( DBusMessageIter& iter, const FileChooserFilter& filter )
{
    ContainerHandle cont( iter, DBUS_TYPE_STRUCT, nullptr );
    dbusAppend( cont, filter.name.c_str() );
    ContainerHandle arr( cont, DBUS_TYPE_ARRAY, "(us)" );
    for ( const auto& glob : filter.globs )
    {
        ContainerHandle globIter( arr, DBUS_TYPE_STRUCT, nullptr );
        dbusAppend( globIter, FileChooserFilter::GlobKind );
        dbusAppend( globIter, glob.c_str() );
    }
    for ( const auto& mime : filter.mimes )
    {
        ContainerHandle mimeIter( arr, DBUS_TYPE_STRUCT, nullptr );
        dbusAppend( mimeIter, FileChooserFilter::MimeKind );
        dbusAppend( mimeIter, mime.c_str() );
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

struct Response
{
    enum Code : uint32_t
    {
        Succeeded = 0,
        UserCancelled = 1,
        Unknown = 2,
    } code;
    std::vector<std::string> uris;
    // choices are not used for now
    std::optional<std::string> currentFilterName; // name only to simplify parsing
};

std::optional<Response> parseResponse( DBusMessage* msg )
{
    Response result;

    DBusMessageIter iter;
    dbus_message_iter_init( msg, &iter );

    if ( dbus_message_iter_get_arg_type( &iter ) != DBUS_TYPE_UINT32 )
        return {};
    dbus_message_iter_get_basic( &iter, &result.code );
    if ( result.code != Response::Succeeded )
        return result;

    dbus_message_iter_next( &iter );

    if ( dbus_message_iter_get_arg_type( &iter ) != DBUS_TYPE_ARRAY )
        return {};
    DBusMessageIter dict;
    dbus_message_iter_recurse( &iter, &dict );
    do
    {
        if ( dbus_message_iter_get_arg_type( &dict ) != DBUS_TYPE_DICT_ENTRY )
            return {};
        DBusMessageIter entry;
        dbus_message_iter_recurse( &dict, &entry );

        if ( dbus_message_iter_get_arg_type( &entry ) != DBUS_TYPE_STRING )
            return {};
        const char* key;
        dbus_message_iter_get_basic( &entry, &key );

        if ( !dbus_message_iter_next( &entry ) )
            break;

        if ( dbus_message_iter_get_arg_type( &entry ) != DBUS_TYPE_VARIANT )
            return {};
        DBusMessageIter varIter;
        dbus_message_iter_recurse( &entry, &varIter );
        if ( strcmp( key, "uris" ) == 0 )
        {
            if ( dbus_message_iter_get_arg_type( &varIter ) != DBUS_TYPE_ARRAY )
                return {};
            DBusMessageIter arrIter;
            dbus_message_iter_recurse( &varIter, &arrIter );
            do
            {
                if ( dbus_message_iter_get_arg_type( &arrIter ) != DBUS_TYPE_STRING )
                    return {};
                const char* uri;
                dbus_message_iter_get_basic( &arrIter, &uri );
                result.uris.emplace_back( uri );
            }
            while ( dbus_message_iter_next( &arrIter ) );
        }
        else if ( strcmp( key, "current_filter" ) == 0 )
        {
            if ( dbus_message_iter_get_arg_type( &varIter ) != DBUS_TYPE_STRUCT )
                return {};
            DBusMessageIter strIter;
            dbus_message_iter_recurse( &varIter, &strIter );

            if ( dbus_message_iter_get_arg_type( &strIter ) != DBUS_TYPE_STRING )
                return {};
            const char* name;
            dbus_message_iter_get_basic( &strIter, &name );
            result.currentFilterName = name;
        }
    }
    while ( dbus_message_iter_next( &dict ) );

    return result;
}

std::optional<Response> waitForResponse( DBusConnection* conn )
{
    while ( true )
    {
        if ( !dbus_connection_read_write( conn, -1 ) )
            return {};

        while ( auto* respMsg = dbus_connection_pop_message( conn ) )
        {
            MR_FINALLY { dbus_message_unref( respMsg ); };
            if ( dbus_message_is_signal( respMsg, "org.freedesktop.portal.Request", "Response" ) )
                return parseResponse( respMsg );
        }
    }
}

char fromHex( char hex )
{
    if ( '0' <= hex && hex <= '9' )
        return hex - '0';
    if ( 'A' <= hex && hex <= 'F' )
        return hex - 'A' + 10;
    if ( 'a' <= hex && hex <= 'f' )
        return hex - 'a' + 10;
    MR_UNREACHABLE
}

std::string percentDecode( std::string_view str )
{
    std::string result;
    size_t cur = 0;
    for ( auto pos = str.find( '%' ); pos != std::string_view::npos; pos = str.find( '%', cur ) )
    {
        result.append( str.substr( cur, pos - cur ) );
        cur = pos + 3;
        if ( str.size() < cur )
            return result;

        const auto* substr = str.data() + pos;
        char decoded = fromHex( substr[1] ) * 0x10 + fromHex( substr[2] );
        result.append( { decoded } );
    }
    result.append( str.substr( cur ) );
    return result;
}

} // namespace

namespace MR::detail
{

constexpr auto* cPortalServiceName = "org.freedesktop.portal.Desktop";
constexpr auto* cPortalObjectPath = "/org/freedesktop/portal/desktop";
constexpr auto* cFileChooserInterface = "org.freedesktop.portal.FileChooser";

bool isPortalFileDialogSupported()
{
    static bool sIsSupported = []
    {
        auto* conn = getConnection();
        if ( !conn )
            return false;

        DBusError err;
        dbus_error_init( &err );
        MR_FINALLY { dbus_error_free( &err ); };

        if ( !dbus_bus_name_has_owner( conn, cPortalServiceName, &err ) )
        {
            spdlog::warn( "XDG Desktop Portal implementation is not found" );
            return false;
        }
        if ( dbus_error_is_set( &err ) )
        {
            spdlog::warn( "DBus error: {}", err.message );
            return false;
        }

        auto* callMsg = dbus_message_new_method_call( cPortalServiceName, cPortalObjectPath, "org.freedesktop.DBus.Properties", "Get" );
        DBusMessageIter argsIter;
        dbus_message_iter_init_append( callMsg, &argsIter );
        dbusAppend( argsIter, cFileChooserInterface );
        dbusAppend( argsIter, "version" );

        auto* reply = dbus_connection_send_with_reply_and_block( conn, callMsg, DBUS_TIMEOUT_INFINITE, &err );
        dbus_connection_flush( conn );
        dbus_message_unref( callMsg );
        if ( !reply || dbus_error_is_set( &err ) )
        {
            spdlog::warn( "Failed to send DBus message: {}", err.message );
            return false;
        }

        DBusMessageIter respIter;
        dbus_message_iter_init( reply, &respIter );
        assert( dbus_message_iter_get_arg_type( &respIter ) == DBUS_TYPE_VARIANT );
        DBusMessageIter varIter;
        dbus_message_iter_recurse( &respIter, &varIter );
        assert( dbus_message_iter_get_arg_type( &varIter ) == DBUS_TYPE_UINT32 );
        uint32_t version;
        dbus_message_iter_get_basic( &varIter, &version );
        spdlog::info( "File Chooser portal found: version {}", version );

        return true;
    } ();
    return sIsSupported;
}

std::vector<std::filesystem::path> runPortalFileDialog( const MR::FileDialog::Parameters& params )
{
    auto* conn = getConnection();
    if ( !conn )
        return {};

    // https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.FileChooser.html
    auto* callMsg = dbus_message_new_method_call( cPortalServiceName, cPortalObjectPath, cFileChooserInterface, params.saveDialog ? "SaveFile" : "OpenFile" );

    const auto parentWindowId = getWindowId( getViewerInstance().window );

    const auto* title = params.multiselect ? "Open Files" : "Open File";
    if ( params.folderDialog )
        title = params.multiselect ? "Open Folders" : "Open Folder";
    else if ( params.saveDialog )
        title = params.multiselect ? "Save Files" : "Save File";

    FileChooserOptions options;
    options.handleToken = "MeshLibFileDialogPortal"; // randomize token?
    options.currentFolder = params.baseFolder;
    for ( const auto& filter : params.filters )
    {
        options.filters.push_back( {
            .name = filter.name,
            .globs = split( filter.extensions, ";" ),
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

    initArgs( callMsg, parentWindowId.c_str(), title, options );

    DBusError err;
    dbus_error_init( &err );
    MR_FINALLY { dbus_error_free( &err ); };

    dbus_connection_send_with_reply_and_block( conn, callMsg, DBUS_TIMEOUT_INFINITE, &err );
    dbus_connection_flush( conn );
    dbus_message_unref( callMsg );
    if ( dbus_error_is_set( &err ) )
    {
        spdlog::warn( "Failed to send DBus message: {}", err.message );
        return {};
    }

    auto resp = waitForResponse( conn );
    if ( !resp )
        return {};
    if ( resp->code != Response::Succeeded )
        return {};

    std::vector<std::filesystem::path> results;
    std::vector<std::string> availExts;
    if ( resp->currentFilterName )
    {
        for ( const auto& filter : options.filters )
        {
            if ( filter.name == *resp->currentFilterName )
            {
                for ( const auto& glob : filter.globs )
                    availExts.emplace_back( glob.substr( 1 ) );
                break;
            }
        }
    }
    for ( const auto& uri : resp->uris )
    {
        // remove 'file://' prefix and revert percent-encoding
        std::filesystem::path path = percentDecode( uri.substr( 7 ) );
        if ( params.saveDialog && !availExts.empty() )
        {
            // make sure the given filename has a correct extension
            const auto ext = utf8string( path.extension() );
            if ( ext.empty() || std::find( availExts.begin(), availExts.end(), ext ) == availExts.end() )
                path.replace_extension( ext + availExts.front() );
        }
        results.emplace_back( path );
    }
    return results;
}

} // namespace MR::detail
#endif
