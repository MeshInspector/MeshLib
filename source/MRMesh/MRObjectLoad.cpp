#include "MRObjectLoad.h"
#include "MRObjectMesh.h"
#include "MRMeshLoad.h"
#include "MRLinesLoad.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRDistanceMapLoad.h"
#include "MRPointsLoad.h"
#include "MRObjectLines.h"
#include "MRObjectPoints.h"
#include "MRDistanceMap.h"
#include "MRObjectDistanceMap.h"
#include "MRStringConvert.h"

namespace MR
{

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

    Vector<Color, VertId> colors;
    auto distanceMapWithParams = DistanceMapLoad::loadRaw( file, callback );
    if ( !distanceMapWithParams.has_value() )
    {
        return tl::make_unexpected( distanceMapWithParams.error() );
    }

    ObjectDistanceMap objectDistanceMap;
    objectDistanceMap.setName( utf8string( file.stem() ) );
    objectDistanceMap.setDistanceMap( std::make_shared<MR::DistanceMap>( std::move( distanceMapWithParams.value().first ) ), std::move( distanceMapWithParams.value().second ) );

    return objectDistanceMap;
}

tl::expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path & folder )
{
    MR_TIMER;

    Object root;

    struct LoadTask
    {
        std::future< tl::expected<ObjectMesh, std::string> > future;
        Object * parent = nullptr;
        LoadTask( std::future< tl::expected<ObjectMesh, std::string> > future, Object * parent ) : future( std::move( future ) ), parent( parent ) { }
    };
    std::vector<LoadTask> loadTasks;

    struct Subfolder
    {
        std::filesystem::path path;
        Object * parent = nullptr;
        Subfolder( std::filesystem::path path, Object * parent ) : path( std::move( path ) ), parent( parent ) { }
    };
    std::vector<Subfolder> subfoldersToLoad{ { folder, &root } };

    while ( !subfoldersToLoad.empty() )
    {
        Subfolder s = std::move( subfoldersToLoad.back() );
        subfoldersToLoad.pop_back();

        std::error_code ec;
	    for ( auto & directoryEntry: std::filesystem::directory_iterator( s.path, ec ) )
        {
            auto path = directoryEntry.path();
            if ( directoryEntry.is_directory( ec ) )
            {
                auto pObj = std::make_shared<Object>();
                pObj->setName( utf8string( path.stem() ) );
                s.parent->addChild( pObj );
                subfoldersToLoad.emplace_back( directoryEntry.path(), pObj.get() );
                continue;
            }
            if ( !directoryEntry.is_regular_file(ec) )
                continue;

            auto ext = utf8string( path.extension() );
            for ( auto & c : ext )
                c = (char) tolower( c );

            loadTasks.emplace_back( 
                std::async( std::launch::async, [path]() { return makeObjectMeshFromFile( path ); } ),
                s.parent
            );
        }
    }

    bool atLeastOneLoaded = false;
    std::string lastError;
    for ( auto & t : loadTasks )
    {
        auto res = t.future.get();
        if ( res.has_value() )
        {            
            t.parent->addChild( std::make_shared<ObjectMesh>( std::move( res.value() ) ) );
            if ( !atLeastOneLoaded )
                atLeastOneLoaded = true;
        }
        else
        {
            lastError = res.error();
        }
    }
    if ( !atLeastOneLoaded )
        return tl::make_unexpected( lastError );

    return root;
}

} //namespace MR
