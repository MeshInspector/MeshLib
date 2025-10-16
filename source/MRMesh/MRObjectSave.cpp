#include "MRObjectSave.h"
#include "MRBitSetParallelFor.h"
#include "MRDistanceMap.h"
#include "MRDistanceMapSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRLinesSave.h"
#include "MRMeshSave.h"
#include "MRObjectDistanceMap.h"
#include "MRObjectLines.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"
#include "MRObjectsAccess.h"
#include "MRPointsSave.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRZip.h"

#include "MRPch/MRJson.h"

namespace
{

using namespace MR;

Mesh mergeToMesh( const Object& object )
{
    Mesh result;
    if ( const auto* objMesh = dynamic_cast<const ObjectMesh*>( &object ) )
    {
        if ( const auto& mesh = objMesh->mesh() )
        {
            result = *mesh;
            result.transform( objMesh->worldXf() );
        }
    }
    for ( const auto& objMesh : getAllObjectsInTree<ObjectMesh>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable ) )
    {
        if ( !objMesh || !objMesh->mesh() )
            continue;

        VertMap vmap;
        result.addMesh( *objMesh->mesh(), nullptr, &vmap );

        const auto xf = objMesh->worldXf();
        for ( const auto v : vmap )
            if ( v.valid() )
                result.points[v] = xf( result.points[v] );
        result.invalidateCaches();
    }
    return result;
}

PointCloud mergeToPoints( const Object& object )
{
    PointCloud result;
    if ( const auto* objPoints = dynamic_cast<const ObjectPoints*>( &object ) )
    {
        if ( const auto& pointCloud = objPoints->pointCloud() )
        {
            result = *pointCloud;
            const auto xf = objPoints->worldXf();
            BitSetParallelFor( result.validPoints, [&] ( const VertId v )
            {
                result.points[v] = xf( result.points[v] );
            } );
            result.invalidateCaches();
        }
    }
    for ( const auto& objPoints : getAllObjectsInTree<ObjectPoints>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable ) )
    {
        if ( !objPoints || !objPoints->pointCloud() )
            continue;

        VertMap vmap;
        result.addPartByMask( result, objPoints->pointCloud()->validPoints, { .src2tgtVerts = &vmap } );

        const auto xf = objPoints->worldXf();
        for ( const auto v : vmap )
            if ( v.valid() )
                result.points[v] = xf( result.points[v] );
        result.invalidateCaches();
    }
    return result;
}

Polyline3 mergeToLines( const Object& object )
{
    Polyline3 result;
    if ( const auto* objLines = dynamic_cast<const ObjectLines*>( &object ) )
    {
        if ( const auto& polyline = objLines->polyline() )
        {
            result = *polyline;
            result.transform( objLines->worldXf() );
        }
    }
    for ( const auto& objLines : getAllObjectsInTree<ObjectLines>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable ) )
    {
        if ( !objLines || !objLines->polyline() )
            continue;

        VertMap vmap;
        result.addPart( *objLines->polyline(), &vmap );

        const auto xf = objLines->worldXf();
        for ( const auto& v : vmap )
            if ( v.valid() )
                result.points[v] = xf( result.points[v] );
        result.invalidateCaches();
    }
    return result;
}

} // namespace

namespace MR
{

namespace ObjectSave
{

Expected<void> toAnySupportedSceneFormat( const Object& object, const std::filesystem::path& file,
                                     ProgressCallback callback )
{
    // NOTE: single-char string literal may break due to the GCC bug:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    auto extension = '*' + toLower( utf8string( file.extension() ) );
    auto saver = SceneSave::getSceneSaver( extension );
    if ( !saver )
        return unexpected( "unsupported file format" );

    return saver( object, file, callback );
}

Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                     ProgressCallback callback )
{
    // NOTE: single-char string literal may break due to the GCC bug:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    const auto extension = '*' + toLower( utf8string( file.extension() ) );
    if ( findFilter( SceneSave::getFilters(), extension ) )
    {
        return toAnySupportedSceneFormat( object, file, callback );
    }
    else if ( findFilter( ObjectSave::getFilters(), extension ) )
    {
        auto saver = ObjectSave::getObjectSaver( extension );
        assert( saver );
        return saver( object, file, callback );
    }
    else if ( findFilter( MeshSave::getFilters(), extension ) )
    {
        const auto mesh = mergeToMesh( object );
        return MeshSave::toAnySupportedFormat( mesh, file, { .progress = callback } );
    }
    else if ( findFilter( PointsSave::getFilters(), extension ) )
    {
        const auto pointCloud = mergeToPoints( object );
        return PointsSave::toAnySupportedFormat( pointCloud, file, { .progress = callback } );
    }
    else if ( findFilter( LinesSave::getFilters(), extension ) )
    {
        const auto polyline = mergeToLines( object );
        return LinesSave::toAnySupportedFormat( polyline, file, { .progress = callback } );
    }
    else if ( findFilter( DistanceMapSave::getFilters(), extension ) )
    {
        const auto objDmaps = getAllObjectsInTree<ObjectDistanceMap>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable );
        if ( objDmaps.empty() )
            return DistanceMapSave::toAnySupportedFormat( {}, file );
        else if ( objDmaps.size() > 1 )
            return unexpected( "Multiple distance maps in the given object" );

        const auto& objDmap = objDmaps.front();
        if ( !objDmap || !objDmap->getDistanceMap() )
            return DistanceMapSave::toAnySupportedFormat( {}, file );

        return DistanceMapSave::toAnySupportedFormat( *objDmap->getDistanceMap(), file, { .xf = &objDmap->getToWorldParameters() } );
    }
    else
    {
        return unexpectedUnsupportedFileExtension();
    }
}

} // namespace ObjectSave

Expected<void> serializeObjectTree( const Object& object, const std::filesystem::path& path,
                                  ProgressCallback progressCb, FolderCallback preCompress )
{
    MR_TIMER;
    if (path.empty())
        return unexpected( "Cannot save to empty path" );

    UniqueTemporaryFolder scenePath( {} );
    if ( !scenePath )
        return unexpected( "Cannot create temporary folder" );

    if ( progressCb && !progressCb( 0.0f ) )
        return unexpected( "Canceled" );

    Json::Value root;
    root["FormatVersion"] = "0.0";
    auto expectedSaveModelFutures = object.serializeRecursive( scenePath, root, 0 );
    if ( !expectedSaveModelFutures.has_value() )
        return unexpected( expectedSaveModelFutures.error() );
    auto & saveModelFutures = expectedSaveModelFutures.value();

    assert( !object.name().empty() );
    auto paramsFile = scenePath / ( object.name() + ".json" );
    if ( !serializeJsonValue( root, paramsFile ) )
        return unexpected( "Cannot write parameters " + utf8string( paramsFile ) );

#ifndef __EMSCRIPTEN__
    if ( !reportProgress( progressCb, 0.1f ) )
        return unexpectedOperationCanceled();

    // wait for all models are saved before making compressed folder
    BitSet inProgress( saveModelFutures.size(), true );
    while ( inProgress.any() )
    {
        for ( auto i : inProgress )
        {
            if ( saveModelFutures[i].wait_for( std::chrono::milliseconds( 200 ) ) != std::future_status::timeout )
                inProgress.reset( i );
        }
        if ( !reportProgress( subprogress( progressCb, 0.1f, 0.9f ), 1.0f - (float)inProgress.count() / inProgress.size() ) )
            return unexpectedOperationCanceled();
    }
#endif

    for ( auto & f : saveModelFutures )
    {
        auto v = f.get();
        if ( !v )
            return v;
    }

    if ( preCompress )
        preCompress( scenePath );

    return compressZip( path, scenePath, {}, nullptr, subprogress( progressCb, 0.9f, 1.0f ) );
}

Expected<void> serializeObjectTree( const Object& object, const std::filesystem::path& path, ProgressCallback progress )
{
    return serializeObjectTree( object, path, std::move( progress ), {} );
}

MR_ADD_SCENE_SAVER_WITH_PRIORITY( IOFilter( "MeshInspector scene (.mru)", "*.mru" ), serializeObjectTree, -1 )

} // namespace MR
