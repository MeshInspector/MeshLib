#include "MROpenObjects.h"
#include <MRMesh/MRObject.h>
#include <MRMesh/MRTimer.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRDirectory.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRZip.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRVoxels/MRDicom.h>

namespace MR
{

Expected<Object> makeObjectTreeFromFolder( const std::filesystem::path & folder, std::string* loadWarn, ProgressCallback callback )
{
    MR_TIMER

    if ( callback && !callback( 0.f ) )
        return unexpected( getCancelMessage( folder ) );

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
            else if ( entry.is_regular_file( ec ) || entry.is_symlink( ec ) )
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
                        break;
                    }
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

    using loadObjResultType = Expected<std::vector<std::shared_ptr<MR::Object>>>;
    // create folders objects
    struct LoadTask
    {
        std::future<loadObjResultType> future;
        Object* parent = nullptr;
        LoadTask( std::future<loadObjResultType> future, Object* parent ) : future( std::move( future ) ), parent( parent ) {}
        bool finished = false;
    };
    std::vector<LoadTask> loadTasks;

    std::atomic_bool loadingCanceled = false;
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
        for ( const FilePathNode& file : node.files )
        {
            loadTasks.emplace_back( std::async( std::launch::async, [&] ()
            {
                return loadObjectFromFile( file.path, loadWarn, [&]( float ){ return !loadingCanceled; } );
            } ), objPtr );
        }
        #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
        if ( node.dicomFolder )
        {
        }
        #endif
    };
    Object result;
    result.setName( utf8string( folder.filename() ) ); // not folder.stem() to preserve "extensions" in folder names
    createFolderObj( filesTree, &result );

    // processing of results
    bool atLeastOneLoaded = false;
    std::unordered_map<std::string, int> allErrors;
    const float taskCount = float( loadTasks.size() );
    int finishedTaskCount = 0;
    std::chrono::system_clock::time_point afterSecond = std::chrono::system_clock::now();
    while ( finishedTaskCount < taskCount )
    {
        afterSecond += +std::chrono::seconds( 1 );
        for ( auto& t : loadTasks )
        {
            if ( !t.future.valid() )
                continue;
            std::future_status status = t.future.wait_until( afterSecond );
            if ( status != std::future_status::ready )
                continue;
            auto res = t.future.get();
            if ( res.has_value() )
            {
                for ( const auto& objPtr : *res )
                {
                    t.parent->addChild( objPtr );
                }
                if ( !atLeastOneLoaded )
                    atLeastOneLoaded = true;
            }
            else
            {
                ++allErrors[res.error()];
            }
            ++finishedTaskCount;
            if ( callback && !callback( finishedTaskCount / taskCount ) )
                loadingCanceled = true;
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
    if ( loadingCanceled )
        return unexpected( getCancelMessage( folder ) );
    if ( !atLeastOneLoaded )
        return unexpected( errorString );

    return result;
}

Expected <Object> makeObjectTreeFromZip( const std::filesystem::path& zipPath, std::string* loadWarn, ProgressCallback callback )
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

    return makeObjectTreeFromFolder( contentsFolder, loadWarn, callback );
}

using ObjectPtr = std::shared_ptr<Object>;

Expected<ObjectPtr> toObjectPtr( Object&& obj )
{
    return std::make_shared<Object>( std::move( obj ) );
}

Expected<ObjectPtr> makeObjectPtrFromZip( const std::filesystem::path& zipPath, std::string* loadWarn, ProgressCallback callback )
{
    return makeObjectTreeFromZip( zipPath, loadWarn, callback ).and_then( toObjectPtr );
}

MR_ADD_SCENE_LOADER( IOFilter( "ZIP files (.zip)","*.zip" ), makeObjectPtrFromZip )

} //namespace MR
