#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"

namespace
{
constexpr float autoVoxelNumber = 5e6f;
}

namespace MR
{

Mesh offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER;

    float voxelSize = params.voxelSize;
    // Compute voxel size if needed
    if ( voxelSize <= 0.0f )
    {
        auto bb = mp.mesh.computeBoundingBox( mp.region );
        auto vol = bb.volume();
        voxelSize = std::cbrt( vol / autoVoxelNumber );
    }

    bool isClosed = mp.mesh.topology.isClosed( mp.region );

    if ( !isClosed )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid
    auto grid = isClosed ?
        // Make level set grid if it is closed
        meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 1,
                        params.callBack ?
                        [params]( float p )
    {
        params.callBack( p * 0.5f );
        return true;
    } : ProgressCallback{} ) :
        // Make distance field grid if it is not closed
        meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 1,
                        params.callBack ?
                             [params]( float p )
    {
        params.callBack( p * 0.5f );
        return true;
    } : ProgressCallback{} );

    // Make offset mesh
    auto newMesh = gridToMesh( grid, voxelSizeVector, offsetInVoxels, params.adaptivity, params.callBack ?
                             [params]( float p )
    {
        params.callBack( 0.5f + p * 0.5f );
        return true;
    } : ProgressCallback{} );

    // For not closed meshes orientation is flipped on back conversion
    if ( !isClosed )
        newMesh.topology.flipOrientation();

    return newMesh;
}

}
