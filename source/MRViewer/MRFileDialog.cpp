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

#ifdef  _WIN32
namespace
{

struct FileDialogParameters : MR::FileParameters
{
    bool folderDialog{false}; // open dialog only
    bool multiselect{true};   // open dialog only
    bool saveDialog{false};   // true for save dialog, false for open
};

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
}
#endif

namespace MR
{

#ifndef _WIN32
std::string convertFiltersToZenity( const MR::IOFilters &filters)
{
    std::string res;
    for(const auto& filter:filters)
    {
        res.append("--file-filter='" + filter.name + " | " + filter.extension + "' ");
    }
    return res;
}
#endif

std::filesystem::path openFileDialog( const FileParameters& params /*= {} */ )
{
    #ifdef  _WIN32
        FileDialogParameters parameters{params};
        parameters.folderDialog = false;
        parameters.multiselect = false;
        parameters.saveDialog = false;
        auto res = windowsDialog( parameters );
        if ( res.size() == 1 )
            return res[0];
        return {};
    #else
#ifndef __EMSCRIPTEN__
        // Gtk has a nasty habit of overriding the locale to "".
        std::string locale = std::setlocale( LC_ALL, nullptr );
        auto kit = Gtk::Application::create();
        std::setlocale( LC_ALL, locale.c_str() );

        Gtk::FileChooserDialog dialog("Open File", Gtk::FILE_CHOOSER_ACTION_OPEN);

        dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
        dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

        for(const auto& filter: params.filters)
        {
            auto filter_text = Gtk::FileFilter::create();
            filter_text->set_name(filter.name);
            filter_text->add_pattern(filter.extension);
            dialog.add_filter(filter_text);
        }

        if ( !params.fileName.empty() )
        {
            dialog.set_current_name( params.fileName );
        }

        std::string res;
        #if defined(__APPLE__)
        int responseId = dialog.run();
        if(responseId == Gtk::RESPONSE_OK)
        {
            res = dialog.get_filename();
        }
        else if(responseId != Gtk::RESPONSE_CANCEL)
        {
            spdlog::warn("GTK failed to get files! Nothing to open.");
        }
        #else // __APPLE__
        dialog.signal_response().connect([&dialog, &res](int responseId)
        {
            if(responseId == Gtk::RESPONSE_OK)
            {
                res = dialog.get_filename();
            }
            else if(responseId != Gtk::RESPONSE_CANCEL)
            {
                spdlog::warn( "GTK failed to get files! Nothing to open." );
            }
            dialog.hide();
        });

        kit->run(dialog);
        #endif // __APPLE__
        return res;
    #else
        return {};
    #endif
#endif
}

std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params /*= {} */ )
{
    #ifdef  _WIN32
       FileDialogParameters parameters{params};
       parameters.folderDialog = false;
       parameters.multiselect = true;
       parameters.saveDialog = false;
       return windowsDialog( parameters );
    #else
#ifndef __EMSCRIPTEN__
        auto kit = Gtk::Application::create();

        Gtk::FileChooserDialog dialog("Open Files", Gtk::FILE_CHOOSER_ACTION_OPEN);
        dialog.set_select_multiple();

        dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
        dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

        for(const auto& filter: params.filters)
        {
            auto filter_text = Gtk::FileFilter::create();
            filter_text->set_name(filter.name);
            filter_text->add_pattern(filter.extension);
            dialog.add_filter(filter_text);
        }

        if ( !params.fileName.empty() )
        {
            dialog.set_current_name( params.fileName );
        }

        std::vector<std::filesystem::path> res;
        #if defined(__APPLE__)
        int responseId = dialog.run();
        if(responseId == Gtk::RESPONSE_OK)
        {
            for (const auto& fileStr : dialog.get_filenames())
            {
                res.push_back(fileStr);
            }
        }
        else if(responseId != Gtk::RESPONSE_CANCEL)
        {
            spdlog::warn("GTK failed to get files! Nothing to open.");
        }
        #else // __APPLE__
        dialog.signal_response().connect([&dialog, &res](int responseId)
        {
            if(responseId == Gtk::RESPONSE_OK)
            {
                for (const auto& fileStr : dialog.get_filenames())
                {
                    res.push_back(fileStr);
                }
            }
            else if(responseId != Gtk::RESPONSE_CANCEL)
            {
                spdlog::warn( "GTK failed to get files! Nothing to open." );
            }
            dialog.hide();
        });

        kit->run(dialog);
        #endif // __APPLE__
        return res;
    #else
        return {};
    #endif
#endif
}

std::filesystem::path openFolderDialog( std::filesystem::path baseFolder /*= {} */ )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();
#ifdef  _WIN32
    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = false;
    parameters.saveDialog = false;
    auto res = windowsDialog( parameters );
    if( res.size() == 1 )
        return res[0];
    return {};
#else
#ifndef __EMSCRIPTEN__
    auto kit = Gtk::Application::create();

    Gtk::FileChooserDialog dialog("Open Files", Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER);
    dialog.set_select_multiple();

    dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
    dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

    std::filesystem::path res;
    #if defined(__APPLE__)
    int responseId = dialog.run();
    if(responseId == Gtk::RESPONSE_OK)
    {
        res = dialog.get_filename();
    }
    else if(responseId != Gtk::RESPONSE_CANCEL)
    {
        spdlog::warn("GTK failed to get files! Nothing to open.");
    }
    #else // __APPLE__
    dialog.signal_response().connect([&dialog, &res](int responseId)
    {
        if(responseId == Gtk::RESPONSE_OK)
        {
            res = dialog.get_filename();
        }
        else if( responseId != Gtk::RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK failed to get files! Nothing to open." );
        }
        dialog.hide();
    });

    kit->run(dialog);
    #endif // __APPLE__

    return res;
#else
    return {};
#endif
#endif
}

std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder /*= {} */ )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();
#ifdef  _WIN32
    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = true;
    parameters.saveDialog = false;
    return windowsDialog( parameters );
#else
#ifndef __EMSCRIPTEN__
    auto kit = Gtk::Application::create();

    Gtk::FileChooserDialog dialog( "Open Files", Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER );
    dialog.set_select_multiple();

    dialog.add_button( Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL );
    dialog.add_button( Gtk::Stock::OPEN, Gtk::RESPONSE_OK );

    std::vector<std::filesystem::path> res;
    #if defined(__APPLE__)
    int responseId = dialog.run();
    if(responseId == Gtk::RESPONSE_OK)
    {
        for ( const auto& fileStr : dialog.get_filenames() )
        {
            res.push_back( fileStr );
        }
    }
    else if(responseId != Gtk::RESPONSE_CANCEL)
    {
        spdlog::warn("GTK failed to get files! Nothing to open.");
    }
    #else // __APPLE__
    dialog.signal_response().connect( [&dialog, &res] ( int responseId )
    {
        if ( responseId == Gtk::RESPONSE_OK )
        {
            for ( const auto& fileStr : dialog.get_filenames() )
            {
                res.push_back( fileStr );
            }
        }
        else if ( responseId != Gtk::RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK failed to get files! Nothing to open." );
        }
        dialog.hide();
    } );

    kit->run( dialog );
    #endif // __APPLE__
    return res;
#else
    return {};
#endif
#endif
}

std::filesystem::path saveFileDialog( const FileParameters& params /*= {} */ )
{
#ifdef  _WIN32
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = true;
    auto res = windowsDialog( parameters );
    if( res.size() == 1 )
        return res[0];
    return {};
#else
#ifndef __EMSCRIPTEN__
    auto kit = Gtk::Application::create();

    Gtk::FileChooserDialog dialog("Save File", Gtk::FILE_CHOOSER_ACTION_SAVE);

    dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
    dialog.add_button(Gtk::Stock::SAVE, Gtk::RESPONSE_OK);

    for ( const auto& filter: params.filters )
    {
        auto filterText = Gtk::FileFilter::create();
        filterText->set_name( filter.name );
        filterText->add_pattern( filter.extension );
        dialog.add_filter( filterText);
    }

    if ( !params.fileName.empty() )
    {
        dialog.set_current_name( params.fileName );
    }

    std::string res;
    auto onResponse = [&dialog, &params, &res]( int responseId )
    {
        if ( responseId == Gtk::RESPONSE_OK )
        {
            using std::filesystem::path;
            auto resPath = path( dialog.get_current_folder() );
            resPath /= path( dialog.get_current_name() );

            if ( !resPath.has_extension() )
            {
                const std::string filterName = dialog.get_filter()->get_name();
                for ( const auto& filter: params.filters )
                {
                    if ( filterName == filter.name )
                    {
                        resPath.replace_extension( filter.extension.substr( 1 ) );
                        break;
                    }
                }
            }

            res = resPath.string();
        }
        else if ( responseId != Gtk::RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK failed to get files! Nothing to open." );
        }
    };

    #if defined( __APPLE__ )
    onResponse( dialog.run() );
    #else // __APPLE__
    dialog.signal_response().connect([&dialog, &onResponse]( int responseId )
    {
        onResponse( responseId );
        dialog.hide();
    });
    kit->run( dialog );
    #endif // __APPLE__
    return res;
#else
    return {};
#endif
#endif
}

}
