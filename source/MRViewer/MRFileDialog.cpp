#include "MRFileDialog.h"

#include "MRViewerFwd.h"
#include "MRColorTheme.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"

#include <GLFW/glfw3.h>

#include <clocale>

#ifndef _WIN32
  #if defined( __APPLE__ )
    #include "MRFileDialogCocoa.h"
  #elif !defined( MRVIEWER_NO_GTK )
    #include <gtk/gtk.h>
  #endif
#else
#  define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef __EMSCRIPTEN__
#  include <GLFW/glfw3native.h>
#endif

#ifdef _WIN32
#  include "MRPch/MRWinapi.h"
#  include <shlobj.h>
#  include <commdlg.h>
#endif

#ifdef __EMSCRIPTEN__
namespace
{
static std::function<void( const std::vector<std::filesystem::path>& )> sDialogFilesCallback;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE int emsOpenFiles( int count, const char** filenames )
{
    if ( !sDialogFilesCallback )
        return 1;
    std::vector<std::filesystem::path> paths( count );
    for ( int i = 0; i < count; ++i )
    {
        paths[i] = MR::pathFromUtf8( filenames[i] );
    }
    sDialogFilesCallback( paths );
    sDialogFilesCallback = {};
    return 0;
}

EMSCRIPTEN_KEEPALIVE int emsSaveFile( const char* filename )
{
    if ( !sDialogFilesCallback )
        return 1;

    std::filesystem::path savePath = std::string( filename );
    sDialogFilesCallback( { savePath } );
    sDialogFilesCallback = {};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( save_file( UTF8ToString( $0 ) ), filename );
#pragma clang diagnostic pop
    return 0;
}

EMSCRIPTEN_KEEPALIVE int emsOpenDirectory( const char* dirname )
{
    if ( !sDialogFilesCallback )
        return 1;

    const std::filesystem::path path = std::string( dirname );
    sDialogFilesCallback( { path } );
    sDialogFilesCallback = {};

    return 0;
}

EMSCRIPTEN_KEEPALIVE void emsFreeFSCallback()
{
    sDialogFilesCallback = {};
}

}

#endif

namespace
{

#if defined( _WIN32 )
std::vector<std::filesystem::path> windowsDialog( const MR::FileDialog::Parameters& params = {} )
{
    std::vector<std::filesystem::path> res;
    //<SnippetRefCounts>
    HRESULT hr = CoInitializeEx( NULL, COINIT_APARTMENTTHREADED |
        COINIT_DISABLE_OLE1DDE );

    std::vector<COMDLG_FILTERSPEC> filters;
    std::vector<std::pair<std::wstring, std::wstring>> filtersWCopy;

    if( SUCCEEDED( hr ) )
    {
        IFileDialog* pFileOpen;

        // Create the FileOpenDialog object.
        hr = CoCreateInstance( params.saveDialog ? CLSID_FileSaveDialog : CLSID_FileOpenDialog,
            NULL, CLSCTX_ALL,
            params.saveDialog ? IID_IFileSaveDialog : IID_IFileOpenDialog,
            reinterpret_cast<void**>(&pFileOpen) );

        if( SUCCEEDED( hr ) )
        {
            auto baseFolder = params.baseFolder;
            if ( baseFolder.empty() )
                baseFolder = MR::FileDialog::getLastUsedDir();
            if( !baseFolder.empty() )
            {
                IShellItem* pItem;
                hr = SHCreateItemFromParsingName( baseFolder.c_str(), NULL, IID_PPV_ARGS( &pItem ) );
                if( SUCCEEDED( hr ) )
                {
                    pFileOpen->SetFolder( pItem );
                    pItem->Release();
                }
            }

            // Set the options on the dialog.
            DWORD dwFlags;
            // Before setting, always get the options first in order
            // not to override existing options.
            pFileOpen->GetOptions( &dwFlags );
            pFileOpen->SetOptions( dwFlags | (params.multiselect ? FOS_ALLOWMULTISELECT : 0) |
                (params.folderDialog ? FOS_PICKFOLDERS : 0) );

            if( !params.filters.empty() )
            {
                unsigned filtersSize = unsigned( params.filters.size() );
                filtersWCopy.resize( filtersSize );
                filters.resize( filtersSize );

                for( unsigned i = 0; i < filtersSize; ++i )
                {
                    const auto& [nameU8, filterU8] = params.filters[i];
                    auto& [nameW, filterW] = filtersWCopy[i];
                    nameW = MR::utf8ToWide( nameU8.c_str() );
                    filterW = MR::utf8ToWide( filterU8.c_str() );

                    filters[i].pszName = nameW.c_str();
                    filters[i].pszSpec = filterW.c_str();
                }

                pFileOpen->SetFileTypes( filtersSize, filters.data() );
                pFileOpen->SetDefaultExtension( L"" );
            }

            if ( !params.fileName.empty() )
            {
                auto filenameW = MR::utf8ToWide( params.fileName.c_str() );
                pFileOpen->SetFileName( filenameW.c_str() );
            }

            // Show the Open dialog box.
            hr = pFileOpen->Show( glfwGetWin32Window( glfwGetCurrentContext() ) );

            // Get the file name from the dialog box.
            if( SUCCEEDED( hr ) )
            {
                if( !params.saveDialog )
                {
                    IFileOpenDialog* fileOpenDialog = (IFileOpenDialog*) pFileOpen;
                    IShellItemArray* pItems;
                    hr = fileOpenDialog->GetResults( &pItems );
                    if( SUCCEEDED( hr ) )
                    {
                        DWORD itemCount;
                        hr = pItems->GetCount( &itemCount );
                        if( SUCCEEDED( hr ) )
                        {
                            res.reserve( itemCount );
                            for( DWORD i = 0; i < itemCount; ++i )
                            {
                                IShellItem* item;
                                hr = pItems->GetItemAt( i, &item );
                                if( SUCCEEDED( hr ) )
                                {
                                    PWSTR pszFilePath;
                                    hr = item->GetDisplayName( SIGDN_FILESYSPATH, &pszFilePath );
                                    if( SUCCEEDED( hr ) )
                                    {
                                        res.push_back( pszFilePath );
                                        CoTaskMemFree( pszFilePath );
                                    }
                                    item->Release();
                                }
                            }
                            pItems->Release();
                            if ( !res.empty() )
                                MR::FileDialog::setLastUsedDir( MR::utf8string( res[0].parent_path() ) );
                        }
                    }
                }
                else
                {
                    IFileSaveDialog* fileSaveDialog = (IFileSaveDialog*) pFileOpen;
                    IShellItem* pItem;
                    hr = fileSaveDialog->GetResult( &pItem );

                    if( SUCCEEDED( hr ) )
                    {
                        PWSTR pszFilePath;
                        hr = pItem->GetDisplayName( SIGDN_FILESYSPATH, &pszFilePath );
                        if( SUCCEEDED( hr ) )
                        {
                            res.push_back( pszFilePath );
                            CoTaskMemFree( pszFilePath );
                        }

                        pItem->Release();
                    }
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
    }

    return res;
}
#else
#ifndef MRVIEWER_NO_GTK
std::tuple<GtkFileChooserAction, std::string> gtkDialogParameters( const MR::FileDialog::Parameters& params )
{
    if ( params.folderDialog )
        return { GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, params.multiselect ? "Open Folders" : "Open Folder" };
    else if ( params.saveDialog )
        return { GTK_FILE_CHOOSER_ACTION_SAVE, "Save File" };
    else
        return { GTK_FILE_CHOOSER_ACTION_OPEN, params.multiselect ? "Open Files" : "Open File" };
}

std::vector<std::filesystem::path> gtkDialog( const MR::FileDialog::Parameters& params = {} )
{
    // Gtk has a nasty habit of overriding the locale to "".s
    std::optional<std::string> localeStr;
    if ( auto locale = std::setlocale( LC_ALL, nullptr ) )
        localeStr = std::string( locale );
    MR_FINALLY {
        if ( localeStr )
            std::setlocale( LC_ALL, localeStr->c_str() );
    };

    if ( !gtk_init_check( NULL, NULL ) )
    {
        spdlog::error( "Failed to initialize GTK+" );
        return {};
    }
    MR_FINALLY {
        while ( gtk_events_pending() )
            gtk_main_iteration();
    };

    auto [action, title] = gtkDialogParameters( params );
    auto* dialog = gtk_file_chooser_dialog_new( title.c_str(), NULL, action, NULL, NULL );
    MR_FINALLY {
        gtk_widget_destroy( dialog );
    };

    gtk_dialog_add_button( GTK_DIALOG( dialog ), params.saveDialog ? "_Save" : "_Open", GTK_RESPONSE_ACCEPT );
    gtk_dialog_add_button( GTK_DIALOG( dialog ), "_Cancel", GTK_RESPONSE_CANCEL );

    auto* chooser = GTK_FILE_CHOOSER( dialog );

    gtk_file_chooser_set_select_multiple( chooser, params.multiselect );

    for ( const auto& filter: params.filters )
    {
        auto* fileFilter = gtk_file_filter_new();
        gtk_file_filter_set_name( fileFilter, filter.name.c_str() );

        size_t separatorPos = 0;
        for (;;)
        {
            auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
            auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );

            gtk_file_filter_add_pattern( fileFilter, ext.c_str() );

            if ( nextSeparatorPos == std::string::npos )
                break;
            separatorPos = nextSeparatorPos + 1;
        }

        gtk_file_chooser_add_filter( chooser, fileFilter ); // the chooser takes ownership of the filter
    }

    const auto currentFolder = params.baseFolder.empty() ?
        MR::FileDialog::getLastUsedDir() : MR::utf8string( params.baseFolder );

    gtk_file_chooser_set_current_folder( chooser, currentFolder.c_str() );

    if ( !params.fileName.empty() )
        gtk_file_chooser_set_current_name( chooser, params.fileName.c_str() );

    if ( params.saveDialog )
        gtk_file_chooser_set_do_overwrite_confirmation( chooser, true );

    std::vector<std::filesystem::path> results;
    auto onResponse = [&] ( int responseId )
    {
        if ( responseId == GTK_RESPONSE_ACCEPT )
        {
            auto* filenames = gtk_file_chooser_get_filenames( chooser );
            MR_FINALLY {
                g_slist_free_full( filenames, g_free );
            };

            for ( auto* node = filenames ; node != NULL; node = node->next )
            {
                const auto* filename = (gchar*)node->data;
                std::filesystem::path filepath( filename );
                if ( params.saveDialog && !filepath.has_extension() )
                {
                    auto* fileFilter = gtk_file_chooser_get_filter( chooser );
                    const std::string filterName = gtk_file_filter_get_name( fileFilter );
                    for ( const auto& filter: params.filters )
                    {
                        if ( filterName == filter.name )
                        {
                            filepath.replace_extension( filter.extensions.substr( 1 ) );
                            break;
                        }
                    }
                }
                results.emplace_back( std::move( filepath ) );
            }

            MR::FileDialog::setLastUsedDir( gtk_file_chooser_get_current_folder( chooser ) );
        }
        else if ( responseId != GTK_RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK dialog failed" );
        }
    };
    onResponse( gtk_dialog_run( GTK_DIALOG( dialog ) ) );
    gtk_widget_hide( dialog );

    return results;
}
#endif
#endif
#ifdef __EMSCRIPTEN__
std::string webAccumFilter( const MR::IOFilters& filters )
{
    std::string accumFilter;
    for (  const auto& filter : filters )
    {
        size_t separatorPos = 0;
        for (;;)
        {
            auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
            auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );
            accumFilter += ( ext.substr( 1 ) + ", " );
            if ( nextSeparatorPos == std::string::npos )
                break;
            separatorPos = nextSeparatorPos + 1;
        }
    }
    accumFilter = accumFilter.substr( 0, accumFilter.size() - 2 );
    return accumFilter;
}
#endif
}

namespace MR
{

std::filesystem::path openFileDialog( const FileParameters& params )
{
    FileDialog::Parameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = false;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif defined( __APPLE__ )
    results = detail::runCocoaFileDialog( parameters );
#elif !defined( MRVIEWER_NO_GTK )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
    {
        if ( !results[0].empty() )
            FileDialogSignals::instance().onOpenFile( results[0] );
        return results[0];
    }
    return {};
}

void openFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFileDialog( params ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
        {
            if ( !paths[0].empty() )
                FileDialogSignals::instance().onOpenFile( paths[0] );
            callback( paths[0] );
        }
    };
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_files_dialog_popup( UTF8ToString( $0 ), $1), accumFilter.c_str(), false);
#pragma clang diagnostic pop
#endif
}

std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params )
{
    FileDialog::Parameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = true;
    parameters.saveDialog = false;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif defined( __APPLE__ )
    results = detail::runCocoaFileDialog( parameters );
#elif !defined( MRVIEWER_NO_GTK )
    results = gtkDialog( parameters );
#endif
    if ( results.empty() )
        spdlog::info( "Open dialog canceled" );
    else
    {
        spdlog::info( "Open dialog returned {} items", results.size() );
        for ( size_t i = 0; i < results.size(); ++i )
        {
            std::error_code ec;
            auto sz = file_size( results[i], ec );
            if ( ec )
                spdlog::info( "  item #{}: {}, access error {}", i, MR::utf8string( results[i] ), ec.message() );
            else
                spdlog::info( "  item #{}: {}, filesize={}", i, MR::utf8string( results[i] ), sz );
        }
        FileDialogSignals::instance().onOpenFiles( results );
    }
    return results;
}

void openFilesDialogAsync( std::function<void( const std::vector<std::filesystem::path>& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFilesDialog( params ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
            FileDialogSignals::instance().onOpenFiles( paths );
        callback( paths );
    };
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_files_dialog_popup( UTF8ToString( $0 ), $1), accumFilter.c_str(), true);
#pragma clang diagnostic pop
#endif
}

std::filesystem::path openFolderDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialog::Parameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = false;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif defined( __APPLE__ )
    results = detail::runCocoaFileDialog( parameters );
#elif !defined( MRVIEWER_NO_GTK )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
    {
        if ( !results[0].empty() )
            FileDialogSignals::instance().onSelectFolder( results[0] );
        return results[0];
    }
    return {};
}

void openFolderDialogAsync( std::function<void ( const std::filesystem::path& )> callback, std::filesystem::path baseFolder )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFolderDialog( std::move( baseFolder ) ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
        {
            if ( !paths[0].empty() )
                FileDialogSignals::instance().onSelectFolder( paths[0] );
            callback( paths[0] );
        }
    };
    (void)baseFolder;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_directory_dialog_popup());
#pragma clang diagnostic pop
#endif
}

std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialog::Parameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = true;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif defined( __APPLE__ )
    results = detail::runCocoaFileDialog( parameters );
#elif !defined( MRVIEWER_NO_GTK )
    results = gtkDialog( parameters );
#endif
    if ( !results.empty() )
        FileDialogSignals::instance().onSelectFolders( results );
    return results;
}

std::filesystem::path saveFileDialog( const FileParameters& params /*= {} */ )
{
    FileDialog::Parameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = true;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif defined( __APPLE__ )
    results = detail::runCocoaFileDialog( parameters );
#elif !defined( MRVIEWER_NO_GTK )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 && !results[0].empty() )
    {
        spdlog::info( "Save dialog returned: {}", MR::utf8string( results[0] ) );
        FileDialog::setLastUsedDir( MR::utf8string( results[0].parent_path() ) );
        FileDialogSignals::instance().onSaveFile( results[0] );
        return results[0];
    }
    spdlog::info( "Save dialog canceled" );
    return {};
}

void saveFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( saveFileDialog( params ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
        {
            if ( !paths[0].empty() )
            {
                FileDialog::setLastUsedDir( MR::utf8string( paths[0].parent_path() ) );
                FileDialogSignals::instance().onSaveFile( paths[0] );
            }
            callback( paths[0] );
        }
    };
    auto filters = params.filters;
    filters.erase( std::remove_if( filters.begin(), filters.end(), [] ( const auto& filter )
    {
        return filter.extensions == "*.*";
    } ), filters.end() );
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( download_file_dialog_popup( UTF8ToString( $0 ), UTF8ToString( $1 ) ), params.fileName.c_str(), accumFilter.c_str() );
#pragma clang diagnostic pop
#endif
}

FileDialogSignals& FileDialogSignals::instance()
{
    static FileDialogSignals inst;
    return inst;
}

namespace FileDialog
{

static const std::string cLastUsedDirKey = "lastUsedDir";

std::string getLastUsedDir()
{
    auto& cfg = Config::instance();
    if ( cfg.hasJsonValue( cLastUsedDirKey ) )
    {
        auto lastUsedDir = cfg.getJsonValue( cLastUsedDirKey );
        if ( lastUsedDir.isString() )
            return lastUsedDir.asString();
    }
    return {};
}

void setLastUsedDir( const std::string& folder )
{
    auto& cfg = Config::instance();
    cfg.setJsonValue( cLastUsedDirKey, folder );
}

} // namespace FileDialog

} // namespace MR
