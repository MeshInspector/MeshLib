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
#include <fstream>
#include <forward_list>

namespace MR
{


/// This class allows progress to be reported from different threads.
/// Unlike progress callback that is passed to \ref ParallelFor, each task may report the progress separately,
/// and the progress displayed to user is not just a number of completed tasks divided by the total number of tasks,
/// but rather a (weighted) average of progresses reported from each task
class ParallelProgressReporter
{
private:

    struct TaskInfo
    {
        float progress = 0.f;
        float weight = 1.f;
    };

public:
    ParallelProgressReporter( const ProgressCallback& cb ):
        cb_( cb )
    {}

    /// Local reporter. It should be passed as a callback to task.
    /// @note One local reporter must not be invoked concurrently.
    struct PerTaskReporter
    {
        bool operator()( float p ) const
        {
            bool res = reporter_->updateTask_( ( p - task_->progress ) * task_->weight );
            task_->progress = p;
            return res;
        }
        ParallelProgressReporter* reporter_ = nullptr;
        TaskInfo* task_ = nullptr;
    };


    /// Add task to the pull
    /// @note This function must not be invoked concurrently.
    /// @return The reporter functor, that could be safely invoked from a different thread
    PerTaskReporter newTask( float weight = 1.f )
    {
        const float totalWeight = totalWeight_;
        progress_ = progress_ * totalWeight / ( totalWeight + weight );
        totalWeight_ += weight;
        return PerTaskReporter( this, &perTaskInfo_.emplace_front( TaskInfo{ .progress = 0.f, .weight = weight } ) );
    }

    /// Actually report the progress. Designed to be invoked in loop until all tasks are completed or until the operation is cancelled
    bool operator()()
    {
        return continue_ = cb_( progress_ );
    }

private:
    /// Invoked from local reporters concurrently.
    bool updateTask_( float delta )
    {
        progress_ += delta / static_cast<float>( totalWeight_ );
        return continue_;
    }

    const ProgressCallback& cb_;

    // progress of each task
    std::forward_list<TaskInfo> perTaskInfo_;

    // sum of the weights of all the tasks
    std::atomic<float> totalWeight_;

    // avg progress for all the tasks
    std::atomic<float> progress_ = 0;

    std::atomic<bool> continue_ = true;
};

Expected<LoadedObject> makeObjectTreeFromFolder( const std::filesystem::path & folder, const ProgressCallback& callback )
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
    // create folders objects
    struct LoadTask
    {
        std::future<loadObjResultType> future;
        Object* parent = nullptr;
        bool dicomFolder = false;
        LoadTask( std::future<loadObjResultType> future, Object* parent, bool dicomFolder = false ) :
            future( std::move( future ) ), parent( parent ), dicomFolder( dicomFolder ) {}
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
            loadTasks.emplace_back( std::async( std::launch::async, [filepath = file.path, cb = cb.newTask()]()
            {
                return loadObjectFromFile( filepath, cb );
            } ), objPtr );
        }
        #if !defined( MESHLIB_NO_VOXELS ) && !defined( MRVOXELS_NO_DICOM )
        if ( node.dicomFolder )
        {
            loadTasks.emplace_back( std::async( std::launch::async, [folder = node.path, cb = cb.newTask( 10.f )]()
            {
                return VoxelsLoad::makeObjectVoxelsFromDicomFolder( folder, cb ).and_then(
                [&]( LoadedObjectVoxels && ld ) -> loadObjResultType
                {
                    return LoadedObjects{ .objs = { ld.obj } };
                } );
            } ), objPtr, true );
        }
        #endif
    };
    LoadedObject res;
    res.obj = std::make_shared<Object>();
    res.obj->setName( utf8string( folder.stem() ) );
    createFolderObj( filesTree, res.obj.get() );

    auto pseudoRoot = std::make_shared<Object>();
    pseudoRoot->addChild( res.obj );

    // processing of results
    bool atLeastOneLoaded = false;
    std::unordered_map<std::string, int> allErrors;
    const float taskCount = float( loadTasks.size() );
    int finishedTaskCount = 0;
    std::chrono::system_clock::time_point afterPause = std::chrono::system_clock::now();
    while ( finishedTaskCount < taskCount )
    {
        afterPause += +std::chrono::milliseconds ( 200 );
        for ( auto& t : loadTasks )
        {
            if ( !cb() )
                loadingCanceled = true;
            if ( !t.future.valid() )
                continue;
            std::future_status status = t.future.wait_until( afterPause );
            if ( status != std::future_status::ready )
                continue;
            auto taskRes = t.future.get();
            if ( taskRes.has_value() )
            {
                if ( t.dicomFolder && taskRes->objs.size() == 1 )
                {
                    t.parent->parent()->addChild( taskRes->objs[0] );
                    t.parent->parent()->removeChild( t.parent );
                }
                else
                {
                    for ( const auto& objPtr : taskRes->objs )
                        t.parent->addChild( objPtr );
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
            ++finishedTaskCount;
        }
    }

    if ( !cb() )
        loadingCanceled = true;

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

    return makeObjectTreeFromFolder( contentsFolder, callback );
}

MR_ADD_SCENE_LOADER( IOFilter( "ZIP files (.zip)","*.zip" ), makeObjectTreeFromZip )

} //namespace MR
