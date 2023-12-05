#include "MRObjectSave.h"
#include "MRDistanceMap.h"
#include "MRDistanceMapSave.h"
#include "MRGltfSerializer.h"
#include "MRLinesSave.h"
#include "MRMeshSave.h"
#include "MRObjectDistanceMap.h"
#include "MRObjectLines.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"
#include "MRObjectVoxels.h"
#include "MRObjectsAccess.h"
#include "MRPointsSave.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRVoxelsSave.h"

namespace
{

using namespace MR;

std::string toLower( std::string str )
{
    for ( auto& ch : str )
        ch = (char)std::tolower( ch );
    return str;
}

bool hasExtension( const MR::IOFilters& filters, const std::string& extension )
{
    return filters.end() != std::find_if( filters.begin(), filters.end(), [&extension] ( const IOFilter& filter )
    {
        return std::string::npos != filter.extensions.find( extension );
    } );
}

Mesh mergeToMesh( const Object& object )
{
    Mesh result;
    for ( const auto& objMesh : getAllObjectsInTree<ObjectMesh>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable ) )
    {
        if ( !objMesh || !objMesh->mesh() )
            continue;

        VertMap vmap;
        result.addPart( *objMesh->mesh(), nullptr, &vmap );

        const auto xf = objMesh->worldXf();
        for ( const auto v : vmap )
            if ( v.valid() )
                result.points[v] = xf( result.points[v] );
    }
    return result;
}

PointCloud mergeToPoints( const Object& object )
{
    PointCloud result;
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
    }
    return result;
}

Polyline3 mergeToLines( const Object& object )
{
    Polyline3 result;
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
    }
    return result;
}

} // namespace

namespace MR::ObjectSave
{

Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                     ProgressCallback callback )
{
    const auto extension = std::string( "*" ) + toLower( utf8string( file.extension().u8string() ) );
    if ( hasExtension( SceneFileFilters, extension ) )
    {
        if ( extension == ".mru" )
            return serializeObjectTree( object, file, callback );
        else if ( extension == ".glb" || extension == ".gltf" )
            return serializeObjectTreeToGltf( object, file, callback );
    }
    else if ( hasExtension( MeshSave::Filters, extension ) )
    {
        const auto mesh = mergeToMesh( object );
        return MeshSave::toAnySupportedFormat( mesh, file, { .progress = callback } );
    }
    else if ( hasExtension( PointsSave::Filters, extension ) )
    {
        const auto pointCloud = mergeToPoints( object );
        return PointsSave::toAnySupportedFormat( pointCloud, file, { .progress = callback } );
    }
    else if ( hasExtension( LinesSave::Filters, extension ) )
    {
        const auto polyline = mergeToLines( object );
        return LinesSave::toAnySupportedFormat( polyline, file, { .progress = callback } );
    }
    else if ( hasExtension( DistanceMapSave::Filters, extension ) )
    {
        const auto objDmaps = getAllObjectsInTree<ObjectDistanceMap>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable );
        if ( objDmaps.empty() )
            return DistanceMapSave::toAnySupportedFormat( file, {} );
        else if ( objDmaps.size() > 1 )
            return unexpected( "Multiple distance maps in the given object" );

        const auto& objDmap = objDmaps.front();
        if ( !objDmap || !objDmap->getDistanceMap() )
            return DistanceMapSave::toAnySupportedFormat( file, {} );

        return DistanceMapSave::toAnySupportedFormat( file, *objDmap->getDistanceMap(), &objDmap->getToWorldParameters() );
    }
    else if ( hasExtension( VoxelsSave::Filters, extension ) )
    {
        const auto objVoxels = getAllObjectsInTree<ObjectVoxels>( const_cast<Object*>( &object ), ObjectSelectivityType::Selectable );
        if ( objVoxels.empty() )
            return VoxelsSave::toAnySupportedFormat( {}, file, callback );
        else if ( objVoxels.size() > 1 )
            return unexpected( "Multiple voxel grids in the given object" );

        const auto& objVoxel = objVoxels.front();
        if ( !objVoxel )
            return VoxelsSave::toAnySupportedFormat( {}, file, callback );

        return VoxelsSave::toAnySupportedFormat( objVoxel->vdbVolume(), file, callback );
    }
    else
    {
        return unexpected( "unsupported file extension" );
    }
}

} // namespace MR::ObjectSave
