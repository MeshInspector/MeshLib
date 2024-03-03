#include "MRObjectSave.h"
#include "MRBitSetParallelFor.h"
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
#include "MRMesh.h"

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
        result.addPart( *objMesh->mesh(), nullptr, &vmap );

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

namespace MR::ObjectSave
{

Expected<void> toAnySupportedSceneFormat( const Object& object, const std::filesystem::path& file,
                                     ProgressCallback callback )
{
    const auto extension = toLower( utf8string( file.extension() ) );
    if ( extension == ".mru" )
        return serializeObjectTree( object, file, callback );
#ifndef MRMESH_NO_GLTF
    else if ( extension == ".glb" || extension == ".gltf" )
        return serializeObjectTreeToGltf( object, file, callback );
#endif
    else
        return unexpected( "unsupported file format" );
}

Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                     ProgressCallback callback )
{
    // NOTE: single-char string literal may break due to the GCC bug:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    const auto extension = '*' + toLower( utf8string( file.extension() ) );
    if ( hasExtension( SceneFileWriteFilters, extension ) )
    {
        return toAnySupportedSceneFormat( object, file, callback );
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
#ifndef MRMESH_NO_VOXEL
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
#endif
    else
    {
        return unexpected( "unsupported file extension" );
    }
}

} // namespace MR::ObjectSave
