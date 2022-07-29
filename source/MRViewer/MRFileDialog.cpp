#include "MRFileDialog.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include <GLFW/glfw3.h>
#include <clocale>

#ifndef _WIN32
  #ifndef __EMSCRIPTEN__
    #include <gtkmm.h>
  #endif
#else
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef __EMSCRIPTEN__
#include <GLFW/glfw3native.h>
#endif

namespace
{

struct FileDialogParameters : MR::FileParameters
{
    bool folderDialog{false}; // open dialog only
    bool multiselect{true};   // open dialog only
    bool saveDialog{false};   // true for save dialog, false for open
};

#ifdef  _WIN32
std::vector<std::filesystem::path> windowsDialog( const FileDialogParameters& params = {} )
{
    std::vector<std::filesystem::path> res;
    //<SnippetRefCounts>
    HRESULT hr = CoInitializeEx( NULL, COINIT_APARTMENTTHREADED |
        COINIT_DISABLE_OLE1DDE );

    COMDLG_FILTERSPEC* filters{ nullptr };
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
            if( !params.baseFolder.empty() )
            {
                IShellItem* pItem;
                hr = SHCreateItemFromParsingName( params.baseFolder.c_str(), NULL, IID_PPV_ARGS( &pItem ) );
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
                filters = new COMDLG_FILTERSPEC[filtersSize];

                for( unsigned i = 0; i < filtersSize; ++i )
                {
                    const auto& [nameU8, filterU8] = params.filters[i];
                    auto& [nameW, filterW] = filtersWCopy[i];
                    nameW = MR::utf8ToWide( nameU8.c_str() );
                    filterW = MR::utf8ToWide( filterU8.c_str() );

                    filters[i].pszName = nameW.c_str();
                    filters[i].pszSpec = filterW.c_str();
                }

                pFileOpen->SetFileTypes( filtersSize, filters );
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

    if( filters )
        delete[] filters;

    return res;
}
#else
#ifndef __EMSCRIPTEN__
std::string gtkDialogTitle( Gtk::FileChooserAction action, bool multiple = false )
{
    switch ( action )
    {
    case Gtk::FILE_CHOOSER_ACTION_OPEN:
        return multiple ? "Open Files" : "Open File";
    case Gtk::FILE_CHOOSER_ACTION_SAVE:
        return "Save File";
    case Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER:
        return multiple ? "Open Folders" : "Open Folder";
    case Gtk::FILE_CHOOSER_ACTION_CREATE_FOLDER:
        return "Save Folder";
    }
    assert( false );
}

std::vector<std::filesystem::path> gtkDialog( const FileDialogParameters& params = {} )
{
    // Gtk has a nasty habit of overriding the locale to "".
    std::string locale = std::setlocale( LC_ALL, nullptr );
    auto kit = Gtk::Application::create();
    std::setlocale( LC_ALL, locale.c_str() );

    Gtk::FileChooserAction action;
    if ( params.folderDialog )
        action = params.saveDialog ? Gtk::FILE_CHOOSER_ACTION_CREATE_FOLDER : Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER;
    else
        action = params.saveDialog ? Gtk::FILE_CHOOSER_ACTION_SAVE : Gtk::FILE_CHOOSER_ACTION_OPEN;
    Gtk::FileChooserDialog dialog( gtkDialogTitle( action, params.multiselect ), action );

    dialog.set_select_multiple( params.multiselect );

    dialog.add_button( Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL );
    dialog.add_button( params.saveDialog ? Gtk::Stock::SAVE : Gtk::Stock::OPEN, Gtk::RESPONSE_ACCEPT );

    for ( const auto& filter: params.filters )
    {
        auto filterText = Gtk::FileFilter::create();
        filterText->set_name( filter.name );
        filterText->add_pattern( filter.extension );
        dialog.add_filter( filterText );
    }

    if ( !params.baseFolder.empty() )
        dialog.set_current_folder( params.baseFolder.string() );
    if ( !params.fileName.empty() )
        dialog.set_current_name( params.fileName );

    std::vector<std::filesystem::path> results;
    auto onResponse = [&] ( int responseId )
    {
        if ( responseId == Gtk::RESPONSE_ACCEPT )
        {
            for ( const auto& filename : dialog.get_filenames() )
            {
                std::filesystem::path filepath( filename );
                if ( params.saveDialog && !filepath.has_extension() )
                {
                    const std::string filterName = dialog.get_filter()->get_name();
                    for ( const auto& filter: params.filters )
                    {
                        if ( filterName == filter.name )
                        {
                            filepath.replace_extension( filter.extension.substr( 1 ) );
                            break;
                        }
                    }
                }
                results.emplace_back( std::move( filepath ) );
            }
        }
        else if ( responseId != Gtk::RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK dialog failed" );
        }
    };
#if defined( __APPLE__ )
    onResponse( dialog.run() );
#else // __APPLE__
    dialog.signal_response().connect([&] ( int responseId )
    {
        onResponse( responseId );
        dialog.hide();
    });
    kit->run( dialog );
#endif // __APPLE__

    return results;
};
#endif
#endif
}

namespace MR
{

std::filesystem::path openFileDialog( const FileParameters& params )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}

std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = true;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    return results;
}

std::filesystem::path openFolderDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = false;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}

std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = true;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    return results;
}

std::filesystem::path saveFileDialog( const FileParameters& params /*= {} */ )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = true;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}

}
