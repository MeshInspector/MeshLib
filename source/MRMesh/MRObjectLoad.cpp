#include "MRObjectLoad.h"
#include "MRObjectMesh.h"
#include "MRMeshLoad.h"
#include "MRLinesLoad.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRDistanceMapLoad.h"
#include "MRImageLoad.h"
#include "MRPointsLoad.h"
#include "MRVoxelsLoad.h"
#include "MRObjectVoxels.h"
#include "MRObjectLines.h"
#include "MRObjectPoints.h"
#include "MRDistanceMap.h"
#include "MRObjectDistanceMap.h"
#include "MRStringConvert.h"
#include "MRIOFormatsRegistry.h"
#include "MRMeshLoadObj.h"
#include "MRSerializer.h"
#include "MRPch/MRSpdlog.h"

#ifndef MRMESH_NO_GLTF
#include "MRGltfSerializer.h"
#endif

namespace MR
{

const IOFilters allFilters = SceneFileFilters
                             | MeshLoad::getFilters()
#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
                             | VoxelsLoad::Filters
#endif
                             | LinesLoad::Filters
                             | PointsLoad::Filters;

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

#if !defined( __EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
tl::expected<std::vector<std::shared_ptr<ObjectVoxels>>, std::string> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;

    auto cb = callback;
    if ( cb )
        cb = [callback] ( float v ) { return callback( v / 3.f ); };
    auto loadRes = VoxelsLoad::fromAnySupportedFormat( file, cb );
    if ( !loadRes.has_value() )
    {
        return tl::make_unexpected( loadRes.error() );
    }
    auto& loadResRef = *loadRes;
    std::vector<std::shared_ptr<ObjectVoxels>> res;
    int size = int( loadResRef.size() );
    for ( int i = 0; i < size; ++i )
    {
        std::shared_ptr<ObjectVoxels> obj = std::make_shared<ObjectVoxels>();
        const std::string name = i > 1 ? fmt::format( "{} {}", utf8string(file.stem()), i) : utf8string(file.stem());
        obj->setName( name );
        int step = 0;
        bool callbackRes = true;
        if ( cb )
            cb = [callback, &i, &step, size, &callbackRes] ( float v )
        {
            callbackRes = callback( ( 1.f + 2 * ( i + ( step + v ) / 2.f ) / size ) / 3.f );
            return callbackRes;
        };

        obj->construct( loadResRef[i], cb );
        if ( cb && !callbackRes )
            return tl::make_unexpected( getCancelMessage( file ) );
        step = 1;
        obj->setIsoValue( ( loadResRef[i].min + loadResRef[i].max ) / 2.f, cb );
        if ( cb && !callbackRes )
            return tl::make_unexpected( getCancelMessage( file ) );
        res.emplace_back( obj );
    }
    
    return res;
}
#endif

tl::expected<std::vector<std::shared_ptr<MR::Object>>, std::string> loadObjectFromFile( const std::filesystem::path& filename,
                                                                                        ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    tl::expected<std::vector<std::shared_ptr<Object>>, std::string> result;

    auto ext = std::string( "*" ) + utf8string( filename.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );   
    
    if ( ext == "*.obj" )
    {
        auto res = MeshLoad::fromSceneObjFile( filename, false, callback );
        if ( res.has_value() )
        {
            std::vector<std::shared_ptr<Object>> objects( res.value().size() );
            auto& resValue = *res;
            for ( int i = 0; i < objects.size(); ++i )
            {
                std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
                if ( resValue[i].name.empty() )
                    objectMesh->setName( utf8string( filename.stem() ) );
                else
                    objectMesh->setName( std::move( resValue[i].name ) );
                objectMesh->select( true );
                objectMesh->setMesh( std::make_shared<Mesh>( std::move( resValue[i].mesh ) ) );
                objectMesh->setUVCoords( std::move( resValue[i].uvCoords ) );
                
                auto image = ImageLoad::fromAnySupportedFormat( resValue[i].pathToTexture );
                if ( image.has_value() )
                {
                    objectMesh->setTexture( { image.value(), FilterType::Linear } );
                    objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                }

                objects[i] = std::dynamic_pointer_cast< Object >( objectMesh );
            }
            result = objects;
        }
        else
            result = tl::make_unexpected( res.error() );
    }
    else if ( std::find_if( SceneFileFilters.begin(), SceneFileFilters.end(), [ext] ( const auto& filter ) { return filter.extension == ext;     } ) != SceneFileFilters.end() )
    {
        const auto objTree = loadSceneFromAnySupportedFormat( filename, callback );
        if ( !objTree.has_value() )
            return tl::make_unexpected( objTree.error() );
        
        result = std::vector( { *objTree } );
        ( *result )[0]->setName( utf8string( filename.stem() ) );
    }
    else
    {
        auto objectMesh = makeObjectMeshFromFile( filename, callback );
        if ( objectMesh.has_value() )
        {
            objectMesh->select( true );
            auto obj = std::make_shared<ObjectMesh>( std::move( *objectMesh ) );
            result = { obj };
        }
        else if ( objectMesh.error() == "Loading canceled" )
        {
            result = tl::make_unexpected( objectMesh.error() );
        }
        else
        {
            result = tl::make_unexpected( objectMesh.error() );

            auto objectPoints = makeObjectPointsFromFile( filename, callback );
            if ( objectPoints.has_value() )
            {
                objectPoints->select( true );
                auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
                result = { obj };
            }
            else if ( result.error() == "unsupported file extension" )
            {
                result = tl::make_unexpected( objectPoints.error() );

                auto objectLines = makeObjectLinesFromFile( filename, callback );
                if ( objectLines.has_value() )
                {
                    objectLines->select( true );
                    auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
                    result = { obj };
                }
                else if ( result.error() == "unsupported file extension" )
                {
                    result = tl::make_unexpected( objectLines.error() );

                    auto objectDistanceMap = makeObjectDistanceMapFromFile( filename, callback );
                    if ( objectDistanceMap.has_value() )
                    {
                        objectDistanceMap->select( true );
                        auto obj = std::make_shared<ObjectDistanceMap>( std::move( objectDistanceMap.value() ) );
                        result = { obj };
                    }
                    else if ( result.error() == "unsupported file extension" )
                    {
                        result = tl::make_unexpected( objectDistanceMap.error() );

#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
                        auto objsVoxels = makeObjectVoxelsFromFile( filename, callback );
                        std::vector<std::shared_ptr<Object>> resObjs;
                        if ( objsVoxels.has_value() )
                        {
                            auto& objsVoxelsRef = *objsVoxels;
                            for ( auto& objPtr : objsVoxelsRef )
                            {
                                objPtr->select( true );
                                resObjs.emplace_back( std::dynamic_pointer_cast< Object >( objPtr ) );
                            }
                            result = resObjs;
                        }
                        else
                            result = tl::make_unexpected( objsVoxels.error() );
#endif

                    }
                }
            }
        }
    }

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}


bool isSupportedFileInSubfolders( const std::filesystem::path& folder )
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


    using loadObjResultType = tl::expected<std::vector<std::shared_ptr<MR::Object>>, std::string>;
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
            pObj->setName( utf8string( folder.path.stem() ) );
            objPtr->addChild( pObj );
            createFolderObj( folder, pObj.get() );
        }
        for ( const FilePathNode& file : node.files )
        {
            loadTasks.emplace_back( std::async( std::launch::async, [&] ()
            {
                return loadObjectFromFile( file.path, [&]( float ){ return !loadingCanceled; } );
            } ), objPtr );
        }
    };
    Object result;
    result.setName( utf8string( folder.stem() ) );
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
        return tl::make_unexpected( getCancelMessage( folder ) );
    if ( !atLeastOneLoaded )
        return tl::make_unexpected( errorString );

    return result;
}

tl::expected<std::shared_ptr<Object>, std::string> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, ProgressCallback callback )
{
    auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto res = tl::make_unexpected( std::string( "unsupported file extension" ) );

    auto itF = std::find_if( SceneFileFilters.begin(), SceneFileFilters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extension == ext;
    } );
    if ( itF == SceneFileFilters.end() )
        return res;

    if ( itF->extension == "*.mru" )
    {
        return deserializeObjectTree( path, {}, callback );
    }
#ifndef MRMESH_NO_GLTF
    else if ( itF->extension == "*.gltf" || itF->extension == "*.glb" )
    {
        return deserializeObjectTreeFromGltf( path, callback );
    }
#endif

    return res;
}

} //namespace MR
