#include "MROpenObjects.h"
#include <MRMesh/MRObject.h>
#include <MRMesh/MRTimer.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRDirectory.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRZip.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRVoxels/MRDicom.h>
#include <MRVoxels/MRObjectVoxels.h>
#include <MRPch/MRSpdlog.h>
#include <MRMesh/MRParallelProgressReporter.h>
#include <MRMesh/MRParallelFor.h>
#include <fstream>
#include "MRPch/MRTBB.h"
#include "MRUnitSettings.h"

namespace MR
{


Expected<LoadedObject> makeObjectTreeFromFolder( const std::filesystem::path & folder, bool dicomOnly, const ProgressCallback& callback )
{
    MR_TIMER;

    if ( callback && !callback( 0.f ) )
        return unexpected( getCancelMessage( folder ) );

    ParallelProgressReporter cb( callback );

    struct FilePathNode
    {
        std::filesystem::path path;
        std::vector<FilePathNode> subfolders;
        std::vector<FilePathNode> files;
        #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
            bool dicomFolder = false;
            VoxelsLoad::DicomStatus dicomStatus = VoxelsLoad::DicomStatusEnum::Invalid;
        #endif

        bool empty() const
        {
            return files.empty() && subfolders.empty()
            #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
                && !dicomFolder
            #endif
                ;
        }
    };

    std::function<FilePathNode( const std::filesystem::path& )> getFilePathNode;
    getFilePathNode = [&getFilePathNode, filters = getAllFilters()] ( const std::filesystem::path& folder )
    {
        FilePathNode node{ .path = folder };
        std::error_code ec;
        for ( auto entry : Directory{ folder, ec } )
        {
            auto path = entry.path();
            if ( entry.is_directory( ec ) )
            {
                node.subfolders.push_back( getFilePathNode( path ) );
            }
            else if ( !node.dicomFolder && ( entry.is_regular_file( ec ) || entry.is_symlink( ec ) ) )
            {
                auto ext = utf8string( path.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
                if ( auto dicomStatus = VoxelsLoad::isDicomFile( path ); dicomStatus != VoxelsLoad::DicomStatusEnum::Invalid ) // unsupported will be reported later
                {
                    node.dicomStatus = dicomStatus;
                    node.dicomFolder = true;
                    node.files.clear();
                }
                else
                #endif
                if ( findFilter( filters, ext ) )
                    node.files.push_back( { .path = path } );
            }
        }
        return node;
    };
    FilePathNode filesTree = getFilePathNode( folder );

    // clear empty folders
    std::function<void( FilePathNode& )> clearEmptySubfolders = {};
    clearEmptySubfolders = [&clearEmptySubfolders] ( FilePathNode& node )
    {
        for ( int i = int( node.subfolders.size() ) - 1; i >= 0; --i )
        {
            clearEmptySubfolders( node.subfolders[i] );
            if ( node.subfolders[i].empty() )
                node.subfolders.erase( node.subfolders.begin() + i );
        }
    };
    clearEmptySubfolders( filesTree );

    using loadObjResultType = Expected<LoadedObjects>;
    struct NodeAndResult
    {
        FilePathNode node;
        Object* parent;
        ProgressCallback cb;
        loadObjResultType result;
    };

    // create folders objects
    std::vector<NodeAndResult> nodes;

    std::function<void( const FilePathNode&, Object* )> createFolderObj = {};
    createFolderObj = [&] ( const FilePathNode& node, Object* objPtr )
    {
        for ( const FilePathNode& folder : node.subfolders )
        {
            auto pObj = std::make_shared<Object>();
            pObj->setName( utf8string( folder.path.filename() ) ); // not path.stem() to preserve "extensions" in folder names
            objPtr->addChild( pObj );
            createFolderObj( folder, pObj.get() );
        }

        if ( !dicomOnly )
            for ( const FilePathNode& file : node.files )
                nodes.push_back( { file, objPtr, cb.newTask() } );

        #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
        if ( node.dicomFolder )
            nodes.push_back( { node, objPtr, cb.newTask( 10.f ) } );
        #endif
    };
    LoadedObject res;
    res.obj = std::make_shared<Object>();
    res.obj->setName( utf8string( folder.stem() ) );
    createFolderObj( filesTree, res.obj.get() );

    if ( nodes.empty() )
    {
        if ( dicomOnly )
            return unexpected( "Could not find any DICOM files." );
        else
            return unexpected( "Error: folder is empty." );
    }

    auto pseudoRoot = std::make_shared<Object>();
    pseudoRoot->addChild( res.obj );

    tbb::task_group group;
    std::atomic<int> completed{ 0 };
    std::atomic<bool> loadingCanceled{ false };
    float dicomScaleFactor = 1.f;
    if ( auto maybeUserScale = UnitSettings::getUiLengthUnit() )
        dicomScaleFactor = getUnitInfo( LengthUnit::meters ).conversionFactor / getUnitInfo( *maybeUserScale ).conversionFactor;

    for ( auto& nodeAndRes : nodes )
    {
        group.run( [&nodeAndRes, &completed, &loadingCanceled, dicomScaleFactor]
        {
            if ( loadingCanceled.load( std::memory_order_relaxed ) )
                return;
            if ( !nodeAndRes.node.dicomFolder )
            {
                nodeAndRes.result = loadObjectFromFile( nodeAndRes.node.path, nodeAndRes.cb );
            }
            #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
            else
            {
                if ( !nodeAndRes.node.dicomStatus )
                    nodeAndRes.result = loadObjResultType( unexpected( fmt::format( "Unsupported DICOM folder: {}", nodeAndRes.node.dicomStatus.reason ) ) );
                else
                    nodeAndRes.result = VoxelsLoad::makeObjectVoxelsFromDicomFolder( nodeAndRes.node.path, nodeAndRes.cb ).and_then(
                        [&, dicomScaleFactor]( LoadedObjects && objs ) -> loadObjResultType
                        {
                            // dicom is always opened in meters, and we can use this information to convert them properly
                            for ( auto& obj : objs.objs )
                                obj->applyScale( dicomScaleFactor );
                            return std::move( objs );
                        } );
            }
            #endif
            completed += 1;
        } );
    }
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    while ( !loadingCanceled && completed < nodes.size() )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds ( 200 ) );
        loadingCanceled = !cb();
        if ( loadingCanceled )
        {
            // do not call group.cancel() since it makes all prallel_for's/parallel_reduce's stop doing what they do normally with unexpecetd concequences
            bool expected = false;
            loadingCanceled.compare_exchange_strong( expected, true, std::memory_order_relaxed );
        }
    }
#endif
    group.wait();

    if ( loadingCanceled )
        return unexpected( getCancelMessage( folder ) );

    // processing of results
    bool atLeastOneLoaded = false;

    struct ErrorInfo
    {
        int count = 0;
        std::filesystem::path path;
    };
    std::unordered_map<std::string, ErrorInfo> allErrors;
    for ( const auto& [node, parent, _, taskRes] : nodes )
    {
        if ( taskRes.has_value() )
        {
            if ( node.dicomFolder && taskRes->objs.size() == 1 )
            {
                parent->parent()->addChild( taskRes->objs[0] );
                parent->parent()->removeChild( parent );
            }
            else
            {
                for ( const auto& objPtr : taskRes->objs )
                    parent->addChild( objPtr );
            }
            if ( !taskRes->warnings.empty() )
                res.warnings += taskRes->warnings;
            if ( !atLeastOneLoaded )
                atLeastOneLoaded = true;
        }
        else
        {
            if ( auto it = allErrors.find( taskRes.error() ); it != allErrors.end() )
            {
                it->second.count += 1;
            }
            else
            {
                std::error_code ec;
                allErrors[taskRes.error()] = { 1, utf8string( std::filesystem::relative( node.path, folder, ec ) ) };
                if ( ec )
                    spdlog::warn( "Filesystem error when trying to obtain {} relative to {}: {}", utf8string( node.path ), utf8string( folder ), ec.message() );
            }
        }
    }

    std::string errorString;
    for ( const auto& error : allErrors )
    {
        errorString += ( errorString.empty() ? "" : "\n" ) + error.first;
        if ( error.second.count > 1 )
        {
            errorString += std::string( " (" ) + std::to_string( error.second.count ) + std::string( ")" );
        }
        else
            errorString += std::string( " (for " ) + utf8string( error.second.path ) + std::string( ")" );
    }

    if ( !atLeastOneLoaded )
        return unexpected( errorString );

    if ( !errorString.empty() )
    {
        spdlog::warn( "Load folder error:\n{}", errorString );
        res.warnings = errorString + '\n' + res.warnings;
    }

    res.obj = pseudoRoot->children()[0];

    return res;
}

Expected<LoadedObject> makeObjectTreeFromZip( const std::filesystem::path& zipPath, const ProgressCallback& callback )
{
    auto tmpFolder = UniqueTemporaryFolder( {} );
    auto contentsFolder = tmpFolder / zipPath.stem();

    std::ifstream in( zipPath, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( zipPath.filename() ) );

    std::error_code ec;
    std::filesystem::create_directory( contentsFolder, ec );
    auto resZip = decompressZip( in, contentsFolder );
    if ( !resZip )
        return unexpected( "ZIP container error: " + resZip.error() );

    return makeObjectTreeFromFolder( contentsFolder, false, callback );
}

MR_ADD_SCENE_LOADER( IOFilter( "ZIP files (.zip)","*.zip" ), makeObjectTreeFromZip )

} //namespace MR
