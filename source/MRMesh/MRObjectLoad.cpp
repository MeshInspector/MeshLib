#include "MRObjectLoad.h"
#include "MRObjectMesh.h"
#include "MRMeshLoad.h"
#include "MRLinesLoad.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRDistanceMapLoad.h"
#include "MRPointsLoad.h"
#include "MRVoxelsLoad.h"
#include "MRObjectLines.h"
#include "MRObjectPoints.h"
#include "MRDistanceMap.h"
#include "MRObjectDistanceMap.h"
#include "MRStringConvert.h"
#include "MRIOFormatsRegistry.h"

namespace MR
{

const IOFilters allFilters = MeshLoad::getFilters();

tl::expected<ObjectMesh, std::string> makeObjectMeshFromFile( const std::filesystem::path & file, ProgressCallback callback )
{
    MR_TIMER;

    Vector<Color, VertId> colors;
    auto mesh = MeshLoad::fromAnySupportedFormat( file, &colors, callback );
    if ( !mesh.has_value() )
    {
        return tl::make_unexpected( mesh.error() );
    }

    ObjectMesh objectMesh;
    objectMesh.setName( utf8string( file.stem() ) );
    objectMesh.setMesh( std::make_shared<MR::Mesh>( std::move( mesh.value() ) ) );
    if ( !colors.empty() )
    {
        objectMesh.setVertsColorMap( std::move( colors ) );
        objectMesh.setColoringType( ColoringType::VertsColorMap );
    }

    return objectMesh;
}

tl::expected<ObjectLines, std::string> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    auto lines = LinesLoad::fromAnySupportedFormat( file, callback );
    if ( !lines.has_value() )
    {
        return tl::make_unexpected( lines.error() );
    }

    ObjectLines objectLines;
    objectLines.setName( utf8string( file.stem() ) );
    objectLines.setPolyline( std::make_shared<MR::Polyline3>( std::move( lines.value() ) ) );

    return objectLines;
}

tl::expected<ObjectPoints, std::string> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    Vector<Color, VertId> colors;
    auto pointsCloud = PointsLoad::fromAnySupportedFormat( file, &colors, callback );
    if ( !pointsCloud.has_value() )
    {
        return tl::make_unexpected( pointsCloud.error() );
    }

    ObjectPoints objectPoints;
    objectPoints.setName( utf8string( file.stem() ) );
    objectPoints.setPointCloud( std::make_shared<MR::PointCloud>( std::move( pointsCloud.value() ) ) );
    if ( !colors.empty() )
    {
        objectPoints.setVertsColorMap( std::move( colors ) );
        objectPoints.setColoringType( ColoringType::VertsColorMap );
    }

    return objectPoints;
}

tl::expected<ObjectDistanceMap, std::string> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    DistanceMapToWorld params;
    auto distanceMap = DistanceMapLoad::fromAnySupportedFormat( file, &params, callback );
    if ( !distanceMap.has_value() )
    {
        return tl::make_unexpected( distanceMap.error() );
    }

    ObjectDistanceMap objectDistanceMap;
    objectDistanceMap.setName( utf8string( file.stem() ) );
    objectDistanceMap.setDistanceMap( std::make_shared<MR::DistanceMap>( std::move( distanceMap.value() ) ), params );

    return objectDistanceMap;
}

bool isAnySupportedFilesInSubfolders( const std::filesystem::path& folder )
{
    std::vector<std::filesystem::path> filesList;
    filesList.push_back( folder );

    while ( !filesList.empty() )
    {
        auto path = filesList[0];
        filesList.erase( filesList.begin() );

        std::error_code ec;
        const std::filesystem::directory_iterator dirEnd;
        for ( auto it = std::filesystem::directory_iterator( path, ec ); !ec && it != dirEnd; it.increment( ec ) )
        {
            auto subpath = it->path();
            if ( it->is_directory( ec ) )
            {
                filesList.push_back( path = subpath );
            }
            else if ( it->is_regular_file( ec ) )
            {
                auto ext = utf8string( subpath.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext.empty() )
                    continue;

                if ( std::find_if( allFilters.begin(), allFilters.end(), [&ext] ( const IOFilter& f )
                    { return f.extension.substr( 1 ) == ext; } ) != allFilters.end() )
                    return true;
            }
        }
    }
    return false;
}

tl::expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path & folder, ProgressCallback callback )
{
    MR_TIMER;

    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( getCancelMessage( folder ) );

    struct FilePathNode
    {
        std::filesystem::path path;
        std::vector<FilePathNode> subfolders;
        std::vector<FilePathNode> files;
    };

    FilePathNode filesTree;
    filesTree.path = folder;


    std::function<void( FilePathNode& )> fillFilesTree = {};
    fillFilesTree = [&fillFilesTree] ( FilePathNode& node )
    {
        std::error_code ec;
        const std::filesystem::directory_iterator dirEnd;
        for ( auto it = std::filesystem::directory_iterator( node.path, ec ); !ec && it != dirEnd; it.increment( ec ) )
        {
            auto path = it->path();
            if ( it->is_directory( ec ) )
            {
                node.subfolders.push_back( { .path = path } );
                fillFilesTree( node.subfolders[node.subfolders.size() - 1] );
            }
            else if ( it->is_regular_file( ec ) )
            {
                auto ext = utf8string( path.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext.empty() )
                    continue;

                if ( std::find_if( allFilters.begin(), allFilters.end(), [&ext] ( const IOFilter& f )
                {
                    return f.extension.substr( 1 ) == ext;
                } ) != allFilters.end() )
                    node.files.push_back( { .path = path } );
            }
        }
    };
    fillFilesTree( filesTree );

    // clear empty folders
    std::function<void( FilePathNode& )> clearEmptySubfolders = {};
    clearEmptySubfolders = [&clearEmptySubfolders] ( FilePathNode& node )
    {
        for ( int i = int( node.subfolders.size() ) - 1; i >= 0; --i )
        {
            clearEmptySubfolders( node.subfolders[i] );
            if ( node.subfolders[i].files.empty() && node.subfolders[i].subfolders.empty() )
                node.subfolders.erase( node.subfolders.begin() + i );
        }
    };
    clearEmptySubfolders( filesTree );

    
    if ( filesTree.subfolders.empty() && filesTree.files.empty() )
        return tl::make_unexpected( std::string( "Error: folder is empty." ) );


    // create folders objects
    struct LoadTask
    {
        std::future< tl::expected<ObjectMesh, std::string> > future;
        Object* parent = nullptr;
        LoadTask( std::future< tl::expected<ObjectMesh, std::string> > future, Object* parent ) : future( std::move( future ) ), parent( parent ) {}
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
            pObj->setName( utf8string( folder.path.stem() ) );
            objPtr->addChild( pObj );
            createFolderObj( folder, pObj.get() );
        }
        for ( const FilePathNode& file : node.files )
        {
            loadTasks.emplace_back( std::async( std::launch::async, [&] ()
            {
                return makeObjectMeshFromFile( file.path, [&]( float ){ return !loadingCanceled; } );
            } ), objPtr );
        }
    };
    Object result;
    result.setName( utf8string( folder.stem() ) );
    createFolderObj( filesTree, &result );

    // processing of results
    bool atLeastOneLoaded = false;
    std::string allErrors;
    const float taskCount = float( loadTasks.size() );
    int finishedTaskCount = 0;
    std::chrono::system_clock::time_point afterSecond = std::chrono::system_clock::now();
    while ( finishedTaskCount < taskCount )
    {
        afterSecond += +std::chrono::seconds( 1 );
        for ( auto& t : loadTasks )
        {
            std::future_status status = t.future.wait_until( afterSecond );
            if ( status != std::future_status::ready )
                continue;
            auto res = t.future.get();
            if ( res.has_value() )
            {
                t.parent->addChild( std::make_shared<ObjectMesh>( std::move( res.value() ) ) );
                if ( !atLeastOneLoaded )
                    atLeastOneLoaded = true;
            }
            else
            {
                allErrors += ( allErrors.empty() ? "" : "\n" ) + res.error();
            }
            ++finishedTaskCount;
            if ( callback && !callback( finishedTaskCount / taskCount ) )
                loadingCanceled = true;
        }
    }
    if ( loadingCanceled )
        return tl::make_unexpected( getCancelMessage( folder ) );
    if ( !atLeastOneLoaded )
        return tl::make_unexpected( allErrors );

    return result;
}

} //namespace MR
