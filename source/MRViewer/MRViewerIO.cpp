#include "MRViewerIO.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRVoxelsSave.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRDistanceMapSave.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRMeshLoadObj.h"
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRViewer/MRAppendHistory.h"

namespace MR
{

VoidOrErrStr saveObjectToFile( const Object& obj, const std::filesystem::path& filename, ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    VoidOrErrStr result;

    if ( auto objPoints = obj.asType<ObjectPoints>() )
    {
        if ( objPoints->pointCloud() )
        {
            const auto& colors = objPoints->getVertsColorMap();
            result = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename,
                                                         colors.empty() ? nullptr : &colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectPoints has no PointCloud in it" ) );
    }
    else if ( auto objLines = obj.asType<ObjectLines>() )
    {
        if ( objLines->polyline() )
        {
            result = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectLines has no Polyline in it" ) );
    }
    else if ( auto objMesh = obj.asType<ObjectMesh>() )
    {
        if ( objMesh->mesh() )
        {
            const Vector<Color, VertId>* colors{ nullptr };
            if ( objMesh->getColoringType() == ColoringType::VertsColorMap )
                colors = &objMesh->getVertsColorMap();

            result = MeshSave::toAnySupportedFormat( *objMesh->mesh(), filename, colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectMesh has no Mesh in it" ) );
    }
    else if ( auto objDistanceMap = obj.asType<ObjectDistanceMap>() )
    {
        if ( auto distanceMap = objDistanceMap->getDistanceMap() )
        {
            result = DistanceMapSave::toAnySupportedFormat( filename, *distanceMap, &objDistanceMap->getToWorldParameters() );
        }
        else
        {
            result = tl::make_unexpected( std::string( "ObjectDistanceMap has no DistanceMap in it" ) );
        }
    }
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
    else if ( auto objVoxels = obj.asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        result = VoxelsSave::toAnySupportedFormat( filename, objVoxels->vdbVolume(), callback );
    }
#endif

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}

}
