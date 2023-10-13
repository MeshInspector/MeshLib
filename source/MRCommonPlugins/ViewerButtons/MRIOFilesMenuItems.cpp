#include "MRIOFilesMenuItems.h"
#include "MRViewer/MRFileDialog.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRGltfSerializer.h"
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
#include "MRMesh/MRGcodeLoad.h"
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
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRViewer/MRSwapRootAction.h"
#include "MRViewer/MRViewerEventsListener.h"
#include "MRPch/MRWasm.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"

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
#ifndef __EMSCRIPTEN__
        auto openDirIt = RibbonSchemaHolder::schema().items.find( "Open directory" );
        if ( openDirIt != RibbonSchemaHolder::schema().items.end() )
            openDirectoryItem_ = std::dynamic_pointer_cast<OpenDirectoryMenuItem>( openDirIt->second.item );
        else
        {
            spdlog::warn( "Cannot find \"Open directory\" menu item for recent files." );
            assert( false );
        }
#endif
        setupListUpdate_();
        connect( &getViewerInstance() );
        // required to be deferred, for valid emscripten static constructors oreder 
        filters_ = MeshLoad::getFilters() | LinesLoad::Filters | PointsLoad::Filters | SceneFileFilters | DistanceMapLoad::Filters | GcodeLoad::Filters;
#ifdef __EMSCRIPTEN__
        std::erase_if( filters_, [] ( const auto& filter )
        {
            return filter.extensions == "*.*";
        } );
#else
#ifndef MRMESH_NO_VOXEL
        filters_ = filters_ | VoxelsLoad::Filters;
#endif
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
            showError( "Unsupported file extension" );
            return;
        }
        getViewerInstance().loadFiles( filenames );
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
        showError( "Unsupported file extension" );
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
        auto headerHeight = viewerRef.framebufferSize.y - sceneBoxSize.y;
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

    getViewerInstance().loadFiles( paths );
    return true;
}

void OpenFilesMenuItem::setupListUpdate_()
{
    if ( recentStoreConnection_.connected() )
        return;

    const auto cutLongFileNames = [this]
    {
        static constexpr size_t fileNameLimit = 75;

        for ( int i = 0; i < dropList_.size(); ++i )
        {
            auto pathStr = utf8string( recentPathsCache_[i] );
            const auto size = pathStr.size();
            if ( size > fileNameLimit )
                pathStr = pathStr.substr( 0, fileNameLimit / 2 ) + " ... " + pathStr.substr( size - fileNameLimit / 2 );

            auto filesystemPath = recentPathsCache_[i];
            dropList_[i] = std::make_shared<LambdaRibbonItem>( pathStr + "##" + std::to_string( i ), 
#ifndef __EMSCRIPTEN__
                [this, filesystemPath] ()
#else
                [filesystemPath] ()
#endif
            {
#ifndef __EMSCRIPTEN__
                std::error_code ec;
                if ( std::filesystem::is_directory( filesystemPath, ec ) )
                {
                    if ( openDirectoryItem_ )
                        openDirectoryItem_->openDirectory( filesystemPath );
                }
                else
#endif
                {
                    getViewerInstance().loadFiles( { filesystemPath } );
                }
            } );
        }
    };

    recentStoreConnection_ = getViewerInstance().recentFilesStore.storageUpdateSignal.connect( [this, cutLongFileNames] ( const FileNamesStack& fileNamesStack ) mutable
    {
        recentPathsCache_ = fileNamesStack;
        dropList_.resize( recentPathsCache_.size() );
        cutLongFileNames();
    } );
    recentPathsCache_ = getViewerInstance().recentFilesStore.getStoredFiles();
    dropList_.resize( recentPathsCache_.size() );
    cutLongFileNames();    
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
            return filter.extensions.find( fileExt ) != std::string::npos;
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

#if !defined(MRMESH_NO_DICOM) && !defined(MRMESH_NO_VOXEL)
void sOpenDICOMs( const std::filesystem::path & directory, const std::string & simpleError )
{
    ProgressBar::orderWithMainThreadPostProcessing( "Open DICOMs", [directory, simpleError, viewer = Viewer::instance()] () -> std::function<void()>
    {
        ProgressBar::nextTask( "Load DICOM Folder" );
        auto loadRes = VoxelsLoad::loadDCMFolderTree( directory, 4, ProgressBar::callBackSetProgress );
        if ( !loadRes.empty() )
        {
            bool anySuccess = std::any_of( loadRes.begin(), loadRes.end(), []( const auto & r ) { return r.has_value(); } );
            std::vector<std::shared_ptr<ObjectVoxels>> voxelObjects;
            ProgressBar::setTaskCount( (int)loadRes.size() * 2 + 1 );
            std::string errors;
            for ( auto & res : loadRes )
            {
                if ( res.has_value() )
                {
                    std::shared_ptr<ObjectVoxels> obj = std::make_shared<ObjectVoxels>();
                    obj->setName( res->name );
                    ProgressBar::nextTask( "Construct ObjectVoxels" );
                    obj->construct( res->vdbVolume, ProgressBar::callBackSetProgress );
                    if ( ProgressBar::isCanceled() )
                    {
                        errors = getCancelMessage( directory );
                        break;
                    }

                    auto bins = obj->histogram().getBins();
                    auto minMax = obj->histogram().getBinMinMax( bins.size() / 3 );
                    ProgressBar::nextTask( "Create ISO surface" );
                    auto isoRes = obj->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                    if ( ProgressBar::isCanceled() )
                    {
                        errors = getCancelMessage( directory );
                        break;
                    }
                    else if ( !isoRes.has_value() )
                    {
                        errors += ( !errors.empty() ? "\n" : "" ) + std::string( isoRes.error() );
                        break;
                    }
                    
                    obj->select( true );
                    obj->setXf( res->xf );
                    voxelObjects.push_back( obj );
                }
                else if ( ProgressBar::isCanceled() )
                {
                    errors = getCancelMessage( directory );
                    break;
                }
                else if ( !anySuccess )
                {
                    if ( simpleError.empty() )
                        errors += ( !errors.empty() ? "\n" : "" ) + res.error();
                    else
                        errors = simpleError;
                }
            }
            return [viewer, voxelObjects, errors, directory] ()
            {
                SCOPED_HISTORY( "Open DICOMs" );
                for ( auto & obj : voxelObjects )
                {
                    AppendHistory<ChangeSceneAction>( "Open DICOMs", obj, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( obj );
                }
                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );

                if ( voxelObjects.size() == 1 )
                {
                    std::filesystem::path scenePath = directory;
                    scenePath += ".mru";
                    getViewerInstance().onSceneSaved( scenePath, false );
                }

                if ( !voxelObjects.empty() )
                    getViewerInstance().recentFilesStore.storeFile( directory );

                if ( !errors.empty() )
                    showError( errors );
            };
        }
        return [] ()
        {
            showError( "Cannot open given folder, find more in log." );
        };
    }, 3 );
}
#endif

bool OpenDirectoryMenuItem::action()
{
    openDirectory( openFolderDialog() );
    return false;
}

void OpenDirectoryMenuItem::openDirectory( const std::filesystem::path& directory ) const
{
    if ( !directory.empty() )
    {
        bool isAnySupportedFiles = isSupportedFileInSubfolders( directory );
        if ( isAnySupportedFiles )
        {
            ProgressBar::orderWithMainThreadPostProcessing( "Open Directory", [directory] ()->std::function<void()>
            {
                auto loadRes = makeObjectTreeFromFolder( directory, ProgressBar::callBackSetProgress );
                if ( loadRes.has_value() )
                {
                    auto obj = std::make_shared<Object>( std::move( *loadRes ) );
                    return[obj, directory]
                    {
                        sSelectRecursive( *obj );
                        AppendHistory<ChangeSceneAction>( "Open Directory", obj, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( obj );
                        getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f } );
                        getViewerInstance().recentFilesStore.storeFile( directory );
                    };
                }
                else
                    return[error = loadRes.error()]
                {
                    showError( error );
                };
            } );
        }
#if !defined(MRMESH_NO_DICOM) && !defined(MRMESH_NO_VOXEL)
        else
        {
            sOpenDICOMs( directory, "No supported files can be open from the directory:\n" + utf8string( directory ) );
        }
#endif
    }
}

#if !defined(MRMESH_NO_DICOM) && !defined(MRMESH_NO_VOXEL)
OpenDICOMsMenuItem::OpenDICOMsMenuItem() :
    RibbonMenuItem( "Open DICOMs" )
{
}

bool OpenDICOMsMenuItem::action()
{
    auto directory = openFolderDialog();
    if ( directory.empty() )
        return false;
    sOpenDICOMs( directory, "No DICOM files can be open from the directory:\n" + utf8string( directory ) );
    return false;
}
#endif
#endif

namespace {

struct SaveInfo
{
    ViewerSettingsManager::ObjType objType;
    const IOFilters& baseFilters;
};

template<typename T>
std::optional<SaveInfo> getSaveInfo( const std::vector<std::shared_ptr<T>> & objs )
{
    std::optional<SaveInfo> res;
    if ( objs.empty() )
        return res;

    auto checkObjects = [&]<class U>( SaveInfo info )
    {
        for ( const auto & obj : objs )
            if ( !dynamic_cast<const U*>( obj.get() ) )
                return false;
        res.emplace( info );
        return true;
    };

    checkObjects.template operator()<ObjectMesh>( { ViewerSettingsManager::ObjType::Mesh, MeshSave::Filters } )
    || checkObjects.template operator()<ObjectLines>( { ViewerSettingsManager::ObjType::Lines, LinesSave::Filters } )
    || checkObjects.template operator()<ObjectPoints>( { ViewerSettingsManager::ObjType::Points, PointsSave::Filters } )
    || checkObjects.template operator()<ObjectDistanceMap>( { ViewerSettingsManager::ObjType::DistanceMap, DistanceMapSave::Filters } )
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
    || checkObjects.template operator()<ObjectVoxels>( { ViewerSettingsManager::ObjType::Voxels, VoxelsSave::Filters } )
#endif
    ;

    return res;
}

} //anonymous namespace

SaveObjectMenuItem::SaveObjectMenuItem() :
    RibbonMenuItem( "Save object" )
{
}

std::string SaveObjectMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>&objs ) const
{
#ifdef __EMSCRIPTEN__
    if ( objs.size() != 1 || !getSaveInfo( objs ) )
        return "Exactly one object of an exportable type must be selected.";
#else
    if ( !getSaveInfo( objs ) )
        return "One or several objects of same exportable type must be selected.";
#endif
    return "";
}

bool SaveObjectMenuItem::action()
{
    const auto objs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( objs.empty() )
        return false;
    const auto optInfo = getSaveInfo( objs );
    if ( !optInfo )
        return false;
    const auto & info = *optInfo;
    const auto objType = info.objType;
    const auto & baseFilters = info.baseFilters;

    int firstFilterNum = 0;
    ViewerSettingsManager* settingsManager = dynamic_cast< ViewerSettingsManager* >( getViewerInstance().getViewportSettingsManager().get() );
    if ( settingsManager )
    {
        const auto& lastExt = settingsManager->getLastExtention( objType );
        if ( !lastExt.empty() )
        {
            for ( int i = 0; i < baseFilters.size(); ++i )
            {
                if ( baseFilters[i].extensions.find( lastExt ) != std::string::npos )
                {
                    firstFilterNum = i;
                    break;
                }
            }
        }
    }

    IOFilters filters;
    if ( firstFilterNum == 0 )
        filters = baseFilters;
    else
    {
        //put filter # firstFilterNum in front of all
        filters = IOFilters( { baseFilters[firstFilterNum] } )
            | IOFilters( baseFilters.begin(), baseFilters.begin() + firstFilterNum )
            | IOFilters( baseFilters.begin() + firstFilterNum + 1, baseFilters.end() );
    }

    saveFileDialogAsync( [objs = std::move( objs ), objType, settingsManager] ( const std::filesystem::path& savePath0 ) mutable
    {
        if ( savePath0.empty() )
            return;
        int objTypeInt = int( objType );
        if ( settingsManager && objTypeInt >= 0 && objTypeInt < int( ViewerSettingsManager::ObjType::Count ) )
            settingsManager->setLastExtention( objType, utf8string( savePath0.extension() ) );
        ProgressBar::orderWithMainThreadPostProcessing( "Save object", [objs = std::move( objs ), savePath0]() -> std::function<void()>
        {
            const auto folder = savePath0.parent_path();
            const auto stem = utf8string( savePath0.stem() );
            // if filename is not the same as object name, then use it as a prefix
            const auto prefix = ( stem == objs[0]->name() ) ? std::string{} : ( stem + "_" );
            const auto ext = utf8string( savePath0.extension() );

            std::vector<std::filesystem::path> savePaths;
            for ( int i = 0; i < objs.size(); ++i )
            {
                std::filesystem::path path = ( objs.size() == 1 ) ? savePath0
                    : ( folder / asU8String( prefix + objs[i]->name() + ext ) );
                const auto sp = subprogress( ProgressBar::callBackSetProgress, float( i ) / objs.size(), float( i + 1 ) / objs.size() );
                auto res = saveObjectToFile( *objs[i], path, { .backupOriginalFile = true, .callback = sp } );
                if ( !res )
                    return [error = std::move( res.error() )] { showError( error ); };
                savePaths.push_back( std::move( path ) );
            }
            return [savePaths = std::move( savePaths )]
            {
                for ( const auto & sp : savePaths )
                    getViewerInstance().recentFilesStore.storeFile( sp );
            };
        } );
    }, { objs[0]->name(), {}, std::move( filters ) } );
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
                    const auto errStr = "Error saving in MRU-format: " + res.error();
                    showError( errStr );
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
            showError( res.error() );
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
        auto res = savePath.extension() == u8".mru" ? serializeObjectTree( root, savePath, ProgressBar::callBackSetProgress ) :
            serializeObjectTreeToGltf( root, savePath, ProgressBar::callBackSetProgress );

        return[savePath, res]()
        {
            if ( res.has_value() )
                getViewerInstance().onSceneSaved( savePath );
            else
            {
                const auto errStr = "Error saving in MRU-format: " + res.error();
                showError( errStr );
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
    StatePlugin( "Capture screenshot" )
{
    CommandLoop::appendCommand( [&] ()
    {
        resolution_ = getViewerInstance().framebufferSize;
    }, CommandLoop::StartPosition::AfterWindowAppear );
}

void CaptureScreenshotMenuItem::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 200.0f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::DragIntValid( "Width", &resolution_.x, 1, 256 );
    ImGui::DragIntValid( "Height", &resolution_.y, 1, 256 );
    if ( UI::button( "Capture", ImVec2( -1, 0 ) ) )
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t( now );
        auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", fmt::localtime( t ) );

        auto savePath = saveFileDialog( { name, {},ImageSave::Filters } );
        if ( !savePath.empty() )
        {
            auto image = Viewer::instanceRef().captureSceneScreenShot( resolution_ );
            auto res = ImageSave::toAnySupportedFormat( image, savePath );
            if ( !res.has_value() )
                showError( "Error saving screenshot: " + res.error() );
        }
    }
    ImGui::EndCustomStatePlugin();
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
                showError( "Error saving screenshot: " + res.error() );
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
    auto image = Viewer::instanceRef().captureSceneScreenShot();

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

#if !defined(MRMESH_NO_DICOM) && !defined(MRMESH_NO_VOXEL)
MR_REGISTER_RIBBON_ITEM( OpenDICOMsMenuItem )
#endif

MR_REGISTER_RIBBON_ITEM( SaveSelectedMenuItem )

MR_REGISTER_RIBBON_ITEM( SaveSceneMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureScreenshotMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureUIScreenshotMenuItem )

MR_REGISTER_RIBBON_ITEM( CaptureScreenshotToClipBoardMenuItem )

#endif

}
#ifdef __EMSCRIPTEN__
#include "MRCommonPlugins/Basic/MRWasmUnavailablePlugin.h"
MR_REGISTER_WASM_UNAVAILABLE_ITEM( OpenDICOMsMenuItem, "Open DICOMs" )
#endif
