#include "MRIOFilesMenuItems.h"
#include "MRViewer/MRFileDialog.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRViewer/MRProgressBar.h"
#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRDistanceMapSave.h"
#include "MRMesh/MRDistanceMapLoad.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewerSettingsManager.h"
#include "MRMesh/MRMeshSaveObj.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRMenu.h"
#include "MRViewer/MRViewerIO.h"
#include "MRViewer/MRSwapRootAction.h"
#include "MRViewer/MRViewerEventsListener.h"
#include "MRPch/MRWasm.h"

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include "MRViewer/MRGladGlfw.h"
#include <GLFW/glfw3native.h>
#endif

#if !defined( _WIN32 ) && !defined( __EMSCRIPTEN__ )
#include <clip/clip.h>
#endif

namespace
{
#ifndef __EMSCRIPTEN__
using namespace MR;
void sSelectRecursive( Object& obj )
{
    obj.select( true );
    for ( auto& child : obj.children() )
        if ( child )
            sSelectRecursive( *child );
}
#endif
}

namespace MR
{

OpenFilesMenuItem::OpenFilesMenuItem() :
    RibbonMenuItem( "Open files" )
{
    type_ = RibbonItemType::ButtonWithDrop;
    // required to be deferred, resent items store to be initialized
    CommandLoop::appendCommand( [&] ()
    {
        setupListUpdate_();
        connect( &getViewerInstance() );
        // required to be deferred, for valid emscripten static constructors oreder 
        filters_ = MeshLoad::getFilters() | LinesLoad::Filters | PointsLoad::Filters | SceneFileFilters | DistanceMapLoad::Filters;
#ifdef __EMSCRIPTEN__
        std::erase_if( filters_, [] ( const auto& filter )
        {
            return filter.extension == "*.*";
        } );
#endif
    } );
}

OpenFilesMenuItem::~OpenFilesMenuItem()
{
    disconnect();
}

bool OpenFilesMenuItem::action()
{
    openFilesDialogAsync( [&] ( const std::vector<std::filesystem::path>& filenames )
    {
        if ( filenames.empty() )
            return;
        if ( !checkPaths_( filenames ) )
        {
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( "Unsupported file extension" );
            return;
        }
        loadFiles_( filenames );
    }, { {}, {}, filters_ } );
    return false;
}

const RibbonMenuItem::DropItemsList& OpenFilesMenuItem::dropItems() const
{
    return dropList_;
}

bool OpenFilesMenuItem::dragDrop_( const std::vector<std::filesystem::path>& paths )
{
    if ( paths.empty() )
        return false;

    if ( !checkPaths_( paths ) )
    {
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            menu->showErrorModal( "Unsupported file extension" );
        return false;
    }

    // if drop to menu scene window -> add objects
    // if drop to viewport -> replace objects
    auto& viewerRef = getViewerInstance();
    SCOPED_HISTORY( "Drag and drop files" );
    auto menu = viewerRef.getMenuPluginAs<RibbonMenu>();
    if ( menu )
    {
        auto sceneBoxSize = menu->getSceneSize();
        auto mousePos = viewerRef.mouseController.getMousePos();
        auto headerHeight = viewerRef.window_height - sceneBoxSize.y;
        if ( mousePos.x > sceneBoxSize.x || mousePos.y < headerHeight )
        {
            auto children = SceneRoot::get().children();
            for ( auto child : children )
            {
                AppendHistory<ChangeSceneAction>( "Remove object", child, ChangeSceneAction::Type::RemoveObject );
                child->detachFromParent();
            }
        }
    }

    loadFiles_( paths );
    return true;
}

void OpenFilesMenuItem::setupListUpdate_()
{
    if ( recentStoreConnection_.connected() )
        return;

    recentStoreConnection_ = getViewerInstance().recentFilesStore.storageUpdateSignal.connect( [this] ( const FileNamesStack& fileNamesStack ) mutable
    {
        recentPathsCache_ = fileNamesStack;
        dropList_.resize( recentPathsCache_.size() );
        for ( int i = 0; i < dropList_.size(); ++i )
        {
            auto pathStr = utf8string( recentPathsCache_[i] );
            auto filesystemPath = recentPathsCache_[i];
            dropList_[i] = std::make_shared<LambdaRibbonItem>( pathStr + "##" + std::to_string( i ), [filesystemPath, this] ()
            {
                loadFiles_( { filesystemPath } );
            } );
        }
    } );
    recentPathsCache_ = getViewerInstance().recentFilesStore.getStoredFiles();
    dropList_.resize( recentPathsCache_.size() );
    
    static constexpr size_t fileNameLimit = 50;

    for ( int i = 0; i < dropList_.size(); ++i )
    {
        auto pathStr = utf8string( recentPathsCache_[i] );
        const auto size = pathStr.size();
        if ( size > fileNameLimit )
            pathStr = pathStr.substr( 0, fileNameLimit / 2 ) + " ... " + pathStr.substr( size - fileNameLimit / 2 );
        
        auto filesystemPath = recentPathsCache_[i];
        dropList_[i] = std::make_shared<LambdaRibbonItem>( pathStr + "##" + std::to_string( i ), [filesystemPath, this] ()
        {
            loadFiles_( { filesystemPath } );
        } );
    }
}

void OpenFilesMenuItem::loadFiles_( const std::vector<std::filesystem::path>& paths )
{
    if ( paths.empty() )
        return;

    ProgressBar::orderWithMainThreadPostProcessing( "Open files", [paths] ()->std::function<void()>
    {
        std::vector<std::filesystem::path> loadedFiles;
        std::vector<std::string> errorList;
        std::vector<std::shared_ptr<Object>> loadedObjects;
        for ( int i = 0; i < paths.size(); ++i )
        {
            const auto& filename = paths[i];
            if ( filename.empty() )
                continue;

            auto res = loadObjectFromFile( filename, [callback = ProgressBar::callBackSetProgress, i, number = paths.size()]( float v )
            {
                return callback( ( i + v ) / number );
            } );
            if ( !res.has_value() )
            {
                errorList.push_back( std::move( res.error() ) );
                continue;
            }

            auto& newObjs = *res;
            bool anyObjLoaded = false;
            for ( auto& obj : newObjs )
            {
                if ( !obj )
                    continue;

                anyObjLoaded = true;
                loadedObjects.push_back( obj );
            }
            if ( anyObjLoaded )
                loadedFiles.push_back( filename );
            else
                errorList.push_back( "No objects found in the file \"" + utf8string( filename ) + "\"" );
        }
        return [loadedObjects, loadedFiles, errorList]
        {
            if ( !loadedObjects.empty() )
            {
                if ( loadedObjects.size() == 1 && std::string( loadedObjects[0]->typeName() ) == std::string( Object::TypeName() ) )
                {
                    AppendHistory<SwapRootAction>( "Load Scene File" );
                    auto newRoot = loadedObjects[0];
                    std::swap( newRoot, SceneRoot::getSharedPtr() );
                    getViewerInstance().onSceneSaved( loadedFiles[0] );
                }
                else
                {
                    std::string historyName = loadedObjects.size() == 1 ? "Open file" : "Open files";
                    SCOPED_HISTORY( historyName );
                    for ( auto& obj : loadedObjects )
                    {
                        AppendHistory<ChangeSceneAction>( "Load File", obj, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( obj );
                    }
                    auto& viewerInst = getViewerInstance();
                    for ( const auto& file : loadedFiles )
                        viewerInst.recentFilesStore.storeFile( file );
                }
                getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f } );
            }
            auto menu = getViewerInstance().getMenuPlugin();
            if ( menu && !errorList.empty() )
            {
                std::string errorAll;
                for ( auto& error : errorList )
                    errorAll += "\n" + error;
                menu->showErrorModal( errorAll.substr( 1 ) );
            }
        };
    } );
}

bool OpenFilesMenuItem::checkPaths_( const std::vector<std::filesystem::path>& paths )
{
    for ( const auto& path : paths )
    {
        std::string fileExt = utf8string( path.extension() );
        for ( auto& c : fileExt )
            c = ( char ) std::tolower( c );
        if ( std::any_of( filters_.begin(), filters_.end(), [&fileExt] ( const auto& filter )
        {
            return filter.extension.substr( 1 ) == fileExt;
        } ) )
            return true;
    }
    return false;
}

#ifndef __EMSCRIPTEN__
OpenDirectoryMenuItem::OpenDirectoryMenuItem() :
    RibbonMenuItem( "Open directory" )
{
}

bool OpenDirectoryMenuItem::action()
{
    auto directory = openFolderDialog();
    if ( !directory.empty() )
    {
        auto container = makeObjectTreeFromFolder( directory );
        if ( container.has_value() && !container->children().empty() )
        {
            auto obj = std::make_shared<Object>( std::move( container.value() ) );
            obj->setName( utf8string( directory.stem() ) );
            sSelectRecursive( *obj );
            AppendHistory<ChangeSceneAction>( "Open directory", obj, ChangeSceneAction::Type::AddObject );
            SceneRoot::get().addChild( obj );
            Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f } );
        }
        else
        {
            ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [directory, viewer = Viewer::instance()] () -> std::function<void()>
            {
                ProgressBar::nextTask( "Load DICOM Folder" );
                auto voxelsObject = VoxelsLoad::loadDCMFolder( directory, 4, ProgressBar::callBackSetProgress );
                if ( voxelsObject && !ProgressBar::isCanceled() )
                {
                    auto bins = voxelsObject->histogram().getBins();
                    auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

                        ProgressBar::nextTask( "Create ISO surface" );
                    voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                    voxelsObject->select( true );
                    return [viewer, voxelsObject] ()
                        {
                        AppendHistory<ChangeSceneAction>( "Open Voxels", voxelsObject, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( voxelsObject );
                        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                    };
                }
                return [viewer]()
                {
                    auto menu = viewer->getMenuPlugin();
                    if ( menu )
                        menu->showErrorModal( "Cannot open given folder, find more in log." );
                };
            }, 2 );
        }
    }
    return false;
}

OpenDICOMsMenuItem::OpenDICOMsMenuItem() :
    RibbonMenuItem( "Open DICOMs" )
{
}

bool OpenDICOMsMenuItem::action()
{
    auto directory = openFolderDialog();
    if ( directory.empty() )
        return false;
    ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [directory, viewer = Viewer::instance()] () -> std::function<void()>
    {
        ProgressBar::nextTask( "Load DICOM Folder" );
        auto voxelObjects = VoxelsLoad::loadDCMFolderTree( directory, 4, ProgressBar::callBackSetProgress );
        if ( !ProgressBar::isCanceled() && !voxelObjects.empty() )
        {
            ProgressBar::setTaskCount( (int)voxelObjects.size() + 1 );
            for ( auto & obj : voxelObjects )
            {
                auto bins = obj->histogram().getBins();
                auto minMax = obj->histogram().getBinMinMax( bins.size() / 3 );

                ProgressBar::nextTask( "Create ISO surface" );
                obj->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                obj->select( true );
            }
            return [viewer, voxelObjects] ()
            {
                for ( auto & obj : voxelObjects )
                {
                    AppendHistory<ChangeSceneAction>( "Open Voxels", obj, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( obj );
                }
                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f }  );
            };
        }
        return [viewer]()
        {
            auto menu = viewer->getMenuPlugin();
            if ( menu )
                menu->showErrorModal( "Cannot open given folder, find more in log." );
        };
    }, 2 );
    return false;
}
#endif

SaveObjectMenuItem::SaveObjectMenuItem() :
    RibbonMenuItem( "Save object" )
{
}

bool SaveObjectMenuItem::action()
{
    auto obj = getDepthFirstObject<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( !obj )
        return false;

    const auto& name = obj->name();
    IOFilters filters;
    IOFilters sortedFilters;
    ViewerSettingsManager* settingsManager = dynamic_cast< ViewerSettingsManager* >( getViewerInstance().getViewportSettingsManager().get() );
    using ObjType = ViewerSettingsManager::ObjType;
    ObjType objType = ObjType::Count;
    auto updateFilters = [&] ( ObjType type, const IOFilters& baseFilters )
    {
        objType = type;
        filters = baseFilters;
        sortedFilters = filters;
        const auto& lastNum = settingsManager->getLastExtentionNum( objType );
        if ( lastNum > 0 && lastNum < filters.size() )
            sortedFilters = IOFilters( { filters[lastNum] } ) | IOFilters( filters.begin(), filters.begin() + lastNum ) | IOFilters( filters.begin() + lastNum + 1, filters.end() );
    };

    if ( settingsManager )
    {
        if ( std::dynamic_pointer_cast< ObjectMesh >( obj ) )
            updateFilters( ObjType::Mesh, MeshSave::Filters );
        if ( std::dynamic_pointer_cast< ObjectLines >( obj ) )
            updateFilters( ObjType::Lines, LinesSave::Filters );
        if ( std::dynamic_pointer_cast< ObjectPoints >( obj ) )
            updateFilters( ObjType::Points, PointsSave::Filters );
        if ( std::dynamic_pointer_cast< ObjectDistanceMap >( obj ) )
            updateFilters( ObjType::DistanceMap, DistanceMapSave::Filters );
#ifndef __EMSCRIPTEN__
        if ( std::dynamic_pointer_cast< ObjectVoxels >( obj ) )
            updateFilters( ObjType::Voxels, VoxelsSave::Filters );
#endif
    }
    else
    {
        if ( std::dynamic_pointer_cast< ObjectMesh >( obj ) )
            sortedFilters = MeshSave::Filters;
        if ( std::dynamic_pointer_cast< ObjectLines >( obj ) )
            sortedFilters = LinesSave::Filters;
        if ( std::dynamic_pointer_cast< ObjectPoints >( obj ) )
            sortedFilters = PointsSave::Filters;
        if ( std::dynamic_pointer_cast< ObjectDistanceMap >( obj ) )
            sortedFilters = DistanceMapSave::Filters;
#ifndef __EMSCRIPTEN__
        if ( std::dynamic_pointer_cast< ObjectVoxels >( obj ) )
            sortedFilters = VoxelsSave::Filters;
#endif
    }

    saveFileDialogAsync( [&] ( const std::filesystem::path& savePath )
    {
        if ( savePath.empty() )
            return;
        int objTypeInt = int( objType );
        if ( settingsManager && objTypeInt >= 0 && objTypeInt < int( ObjType::Count ) )
        {
            const auto extention = '*' + utf8string( savePath.extension() );
            auto findRes = std::find_if( filters.begin(), filters.end(), [&extention] ( const IOFilter& elem )
            {
                return elem.extension == extention;
            } );
            if ( findRes != filters.end() )
                settingsManager->setLastExtentionNum( objType, int( findRes - filters.begin() ) );
            else
                settingsManager->setLastExtentionNum( objType, 0 );
        }
        ProgressBar::orderWithMainThreadPostProcessing( "Save object", [savePath]
        {
            std::optional<std::filesystem::path> copyPath{};
            std::error_code ec;
            std::string copySuffix = ".tmpcopy";
            if ( std::filesystem::is_regular_file( savePath, ec ) )
            {
                copyPath = savePath.string() + copySuffix;
                std::filesystem::copy_file( savePath, copyPath.value(), ec );
            }
            if ( ec )
                spdlog::error( ec.message() );

            auto obj = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected )[0];
            auto res = saveObjectToFile( *obj, savePath, ProgressBar::callBackSetProgress );

            std::function<void()> fnRes = [savePath]
            {
                getViewerInstance().recentFilesStore.storeFile( savePath );
            };
            if ( !res.has_value() )
            {
                fnRes = [error = res.error(), savePath, copyPath]
                {
                    std::error_code ec;
                    std::filesystem::remove( savePath, ec );
                    if ( ec )
                        spdlog::error( ec.message() );
                    if ( copyPath.has_value() )
                    {
                        std::filesystem::rename( copyPath.value(), savePath, ec );
                        if ( ec )
                            spdlog::error( ec.message() );
                    }
                    if ( auto menu = getViewerInstance().getMenuPlugin() )
                        menu->showErrorModal( error );
                };
            }
            else if ( copyPath.has_value() )
            {
                fnRes = [copyPathValue = copyPath.value(), savePath]
                {
                    std::error_code ec;
                    std::filesystem::remove( copyPathValue, ec );
                    if ( ec )
                        spdlog::error( ec.message() );

                    getViewerInstance().recentFilesStore.storeFile( savePath );
                };
            }
            return fnRes;
        } );
    }, { name, {}, sortedFilters } );
    return false;
}


SaveSelectedMenuItem::SaveSelectedMenuItem():
    RibbonMenuItem( "Save selected" )
{
}

void removeUnselectedRecursive( Object& obj )
{
    if ( obj.isAncillary() )
    {
        obj.detachFromParent();
        return;
    }
    auto children = obj.children(); // copy because children can be removed
    bool selected = obj.isSelected();
    if ( children.empty() && !selected )
    {
        obj.detachFromParent();
        return;
    }
    for ( auto& child : children )
        removeUnselectedRecursive( *child );
    if ( selected )
        return;
    if ( obj.children().empty() ) // children could be removed so double check
        obj.detachFromParent();
}

bool SaveSelectedMenuItem::action()
{
    auto selectedMeshes = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    auto selectedObjs = getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    
    auto filters = SceneFileFilters;
    // allow obj format only if all selected objects are meshes
    if ( selectedMeshes.size() == selectedObjs.size() )
        filters = SceneFileFilters | IOFilters{ IOFilter{"OBJ meshes (.obj)","*.obj"} };
    
    auto savePath = saveFileDialog( { {},{},filters } );
    if ( savePath.empty() )
        return false;

    auto ext = savePath.extension().u8string();
    for ( auto& c : ext )
        c = ( char ) tolower( c );

    if ( ext == u8".mru" )
    {
        auto rootShallowClone = SceneRoot::get().shallowCloneTree();
        auto children = rootShallowClone->children();
        for ( auto& child : children )
            removeUnselectedRecursive( *child );

        ProgressBar::orderWithMainThreadPostProcessing( "Saving selected", [savePath, rootShallowClone, viewer = Viewer::instance()]()->std::function<void()>
        {
            auto res = serializeObjectTree( *rootShallowClone, savePath, ProgressBar::callBackSetProgress );
            if ( !res.has_value() )
                spdlog::error( res.error() );

            return[savePath, viewer, res]()
            {
                if ( res.has_value() )
                    viewer->recentFilesStore.storeFile( savePath );
                else
                {
                    if ( auto menu = getViewerInstance().getMenuPlugin() )
                        menu->showErrorModal( "Error saving in MRU-format" );
                    spdlog::error( res.error() );
                }
            };
        } );
    }
    else if ( ext == u8".obj" )
    {
        std::vector<MeshSave::NamedXfMesh> objs;
        for ( auto obj : selectedMeshes )
            objs.push_back( MeshSave::NamedXfMesh{ obj->name(),obj->worldXf(),obj->mesh() } );

        auto res = MeshSave::sceneToObj( objs, savePath );
        if ( !res.has_value() )
        {
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( res.error() );
            spdlog::error( res.error() );
        }
        else
            getViewerInstance().recentFilesStore.storeFile( savePath );
    }
    return false;
}

SaveSceneAsMenuItem::SaveSceneAsMenuItem( const std::string& pluginName ) :
    RibbonMenuItem( pluginName )
{
}

void SaveSceneAsMenuItem::saveScene_( const std::filesystem::path& savePath )
{
    ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [savePath, &root = SceneRoot::get()]()->std::function<void()>
    {
        auto res = serializeObjectTree( root, savePath, ProgressBar::callBackSetProgress );

        return[savePath, res]()
        {
            if ( res.has_value() )
                getViewerInstance().onSceneSaved( savePath );
            else
            {
                spdlog::error( res.error() );
                if ( auto menu = getViewerInstance().getMenuPlugin() )
                    menu->showErrorModal( "Error saving in MRU-format" );
            }
        };
    } );
}

bool SaveSceneAsMenuItem::action()
{
    saveFileDialogAsync( [&] ( const std::filesystem::path& savePath )
    {
        if ( !savePath.empty() )
            saveScene_( savePath );
    }, { {}, {}, SceneFileFilters } );
    return false;
}

SaveSceneMenuItem::SaveSceneMenuItem() :
    SaveSceneAsMenuItem( "Save Scene" )
{
}

bool SaveSceneMenuItem::action()
{   
    auto savePath = SceneRoot::getScenePath();
    if ( savePath.empty() )
        savePath = saveFileDialog( { {}, {},SceneFileFilters } );
    if ( !savePath.empty() )
        saveScene_( savePath );
    return false;
}

CaptureScreenshotMenuItem::CaptureScreenshotMenuItem():
    RibbonMenuItem( "Capture screenshot" )
{
}

bool CaptureScreenshotMenuItem::action()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t( now );
    auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", fmt::localtime( t ) );

    auto savePath = saveFileDialog( { name, {},ImageSave::Filters } );
    if ( !savePath.empty() )
    {
        auto bounds = Viewer::instanceRef().getViewportsBounds();
        auto image = Viewer::instanceRef().captureScreenShot( Vector2i( bounds.min ), Vector2i( bounds.max - bounds.min ) );
        auto res = ImageSave::toAnySupportedFormat( image, savePath );
        if ( !res.has_value() )
        {
            spdlog::warn( "Error saving screenshot: {}", res.error() );
            if ( auto menu = getViewerInstance().getMenuPlugin() )
                menu->showErrorModal( res.error() );
        }
    }
    return false;
}

CaptureUIScreenshotMenuItem::CaptureUIScreenshotMenuItem():
    RibbonMenuItem("Capture UI screenshot")
{
}

bool CaptureUIScreenshotMenuItem::action()
{
    getViewerInstance().captureUIScreenShot( [] ( const Image& image )
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t( now );
        auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", fmt::localtime( t ) );

        auto savePath = saveFileDialog( { name, {},ImageSave::Filters});
        if ( !savePath.empty() )
        {
            auto res = ImageSave::toAnySupportedFormat( image, savePath );
            if ( !res.has_value() )
            {
                spdlog::warn( "Error saving screenshot: {}", res.error() );
                if ( auto menu = getViewerInstance().getMenuPlugin() )
                    menu->showErrorModal( res.error() );
            }
        }
    } );
    return false;
}

CaptureScreenshotToClipBoardMenuItem::CaptureScreenshotToClipBoardMenuItem() :
    RibbonMenuItem( "Screenshot to clipboard" )
{
}

std::string CaptureScreenshotToClipBoardMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
#if defined( __EMSCRIPTEN__ )
    return "This function is not currently supported in web edition";
#else
    return "";
#endif
}

bool CaptureScreenshotToClipBoardMenuItem::action()
{
#ifndef __EMSCRIPTEN__
    auto bounds = Viewer::instanceRef().getViewportsBounds();
    auto image = Viewer::instanceRef().captureScreenShot( Vector2i( bounds.min ), Vector2i( bounds.max - bounds.min ) );

#if defined( _WIN32 )
    auto hwnd = glfwGetWin32Window( getViewerInstance().window );
    auto hdc = GetDC( hwnd );

    HDC memDC = CreateCompatibleDC( hdc );
    HBITMAP memBM = CreateCompatibleBitmap( hdc, image.resolution.x, image.resolution.y );
    SelectObject( memDC, memBM );

    BITMAPINFOHEADER bmih;
    bmih.biSize = sizeof( BITMAPINFOHEADER );
    bmih.biBitCount = 32;
    bmih.biClrImportant = 0;
    bmih.biClrUsed = 0;
    bmih.biCompression = BI_RGB;
    bmih.biHeight = image.resolution.y;
    bmih.biWidth = image.resolution.x;
    bmih.biPlanes = 1;
    bmih.biSizeImage = 0;
    bmih.biXPelsPerMeter = 0;
    bmih.biYPelsPerMeter = 0;

    BITMAPINFO bmpInfo;
    bmpInfo.bmiHeader = bmih;

    std::vector<uint8_t> buffer( 4 * image.resolution.x * image.resolution.y, 0 );
    for ( int y = 0; y < image.resolution.y; ++y )
    {
        int inputShift = y * image.resolution.x;
        int indShift = inputShift * 4;
        for ( int x = 0; x < image.resolution.x; ++x )
        {
            const Color& c = image.pixels[inputShift + x];
            auto ind = indShift + x * 4;
            buffer[ind + 0] = c.b;
            buffer[ind + 1] = c.g;
            buffer[ind + 2] = c.r;
            buffer[ind + 3] = c.a;
        }
    }

    auto res = SetDIBits( memDC, memBM, 0, image.resolution.y, buffer.data(), &bmpInfo, DIB_RGB_COLORS );
    if ( res )
    {
        res = OpenClipboard( hwnd );
        if ( res )
        {
            res = EmptyClipboard();
            if ( res )
            {
                auto hnd = SetClipboardData( CF_BITMAP, memBM );
                if ( !hnd )
                    spdlog::error( "Write screenshot to clipboard failed" );
            }
            else
                spdlog::error( "Cannot empty clipboard" );

            res = CloseClipboard();
            if ( !res )
                spdlog::error( "Cannot close clipboard" );
        }
        else
            spdlog::error( "Cannot open clipboard" );
    }
    else
        spdlog::error( "Make image for clipboard failed" );
    
    res = DeleteObject( memBM );
    if ( !res )
        spdlog::error( "Cannot clear image for clipboard" );
    res = DeleteDC( memDC );
    if ( !res )
        spdlog::error( "Cannot clear compatible device context" );
#elif defined( __APPLE__ )
    clip::image_spec image_spec;
    image_spec.width = image.resolution.x;
    image_spec.height = image.resolution.y;
    image_spec.bits_per_pixel = 32;
    image_spec.bytes_per_row = image.resolution.x * 4;
    image_spec.red_mask = 0x000000ff;
    image_spec.green_mask = 0x0000ff00;
    image_spec.blue_mask = 0x00ff0000;
    image_spec.alpha_mask = 0xff000000;
    image_spec.red_shift = 0;
    image_spec.green_shift = 8;
    image_spec.blue_shift = 16;
    image_spec.alpha_shift = 24;

    // flip image
    for (auto y1 = 0, y2 = image.resolution.y - 1; y1 < y2; y1++, y2--)
        for (auto x = 0, width = image.resolution.x; x < width; x++)
            std::swap(image.pixels[y1 * width + x], image.pixels[y2 * width + x]);

    clip::image img( image.pixels.data(), image_spec );
    clip::set_image( img );
#else
    std::ostringstream oss;
    auto res = MR::ImageSave::toPng( image, oss );
    if ( !res.has_value() )
    {
        spdlog::error( "Cannot convert screenshot" );
        return false;
    }
    auto image_data = oss.str();

    clip::lock lock;
    if ( !lock.clear() )
    {
        spdlog::error( "Cannot clear clipboard" );
        return false;
    }
    if ( !lock.set_data( clip::register_format( "image/png" ), image_data.data(), image_data.size() ) )
    {
        spdlog::error( "Write screenshot to clipboard failed" );
        return false;
    }
#endif
#endif
    return false;
}

MR_REGISTER_RIBBON_ITEM( OpenFilesMenuItem )

MR_REGISTER_RIBBON_ITEM( SaveObjectMenuItem )

MR_REGISTER_RIBBON_ITEM( SaveSceneAsMenuItem )

#ifndef __EMSCRIPTEN__

MR_REGISTER_RIBBON_ITEM( OpenDirectoryMenuItem )

MR_REGISTER_RIBBON_ITEM( OpenDICOMsMenuItem )

MR_REGISTER_RIBBON_ITEM( SaveSelectedMenuItem )

MR_REGISTER_RIBBON_ITEM( SaveSceneMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureScreenshotMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureUIScreenshotMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureScreenshotToClipBoardMenuItem )

#endif

}
