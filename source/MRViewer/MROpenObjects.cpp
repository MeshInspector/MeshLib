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
    MR_TIMER

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

                if ( ext.empty() )
                    continue;

                #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
                if ( ext == ".dcm" && VoxelsLoad::isDicomFile( path ) )
                {
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

    if ( filesTree.empty() )
        return unexpected( std::string( "Error: folder is empty." ) );

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

    auto pseudoRoot = std::make_shared<Object>();
    pseudoRoot->addChild( res.obj );

    tbb::task_group group;
    std::atomic<int> completed;
    bool loadingCanceled = false;
    const float dicomScaleFactor = UnitSettings::getUiLengthUnit()
        .transform( [] ( LengthUnit u ) { return getUnitInfo( LengthUnit::meters ).conversionFactor / getUnitInfo( u ).conversionFactor; } )
        .value_or( 1.f );
    for ( auto& nodeAndRes : nodes )
    {
        group.run( [&nodeAndRes, &completed, dicomScaleFactor] {
            if ( !nodeAndRes.node.dicomFolder )
            {
                nodeAndRes.result = loadObjectFromFile( nodeAndRes.node.path, nodeAndRes.cb );
            }
            #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
            else
            {
                nodeAndRes.result = VoxelsLoad::makeObjectVoxelsFromDicomFolder( nodeAndRes.node.path, nodeAndRes.cb ).and_then(
                    [&, dicomScaleFactor]( LoadedObjectVoxels && ld ) -> loadObjResultType
                    {
                        // dicom is always opened in meters, and we can use this information to convert them properly
                        ld.obj->applyScale( dicomScaleFactor );
                        return LoadedObjects{ .objs = { ld.obj } };
                    } );
            }
            #endif
            completed += 1;
        } );
    }

    while ( !loadingCanceled && completed < nodes.size() )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds ( 200 ) );
        loadingCanceled = !cb();
    }
    group.wait();

    if ( loadingCanceled )
        return unexpected( getCancelMessage( folder ) );

    // processing of results
    bool atLeastOneLoaded = false;
    std::unordered_map<std::string, int> allErrors;
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
            ++allErrors[taskRes.error()];
        }
    }

    std::string errorString;
    for ( const auto& error : allErrors )
    {
        errorString += ( errorString.empty() ? "" : "\n" ) + error.first;
        if ( error.second > 1 )
        {
            errorString += std::string( " (" ) + std::to_string( error.second ) + std::string( ")" );
        }
    }

    if ( !errorString.empty() )
        spdlog::warn( "Load folder error:\n{}", errorString );
    if ( !atLeastOneLoaded )
        return unexpected( errorString );

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
