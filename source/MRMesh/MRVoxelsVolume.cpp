#include "MRVoxelsVolume.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRVDBConversions.h"
#include "MRBoolean.h"
#include "MRFloatGrid.h"

namespace MR
{

float voxelizeAndComputeVolume( const std::vector<std::shared_ptr<Mesh>>& meshes, const AffineXf3f& xf, const Vector3f& voxelSize )
{
    if ( meshes.empty() )
        return 0.0f;
    std::vector<FloatGrid> grids( meshes.size() );
    int goodGridsCounter = 0;
    for ( int i = 0; i < meshes.size(); ++i )
    {
        if ( meshes[i] )
        {
            assert( meshes[i]->topology.isClosed() );
            grids[i] = meshToLevelSet( *meshes[i], xf, voxelSize, 2.0f );
            ++goodGridsCounter;
        }
    }
    if ( goodGridsCounter == 0 )
        return 0.0f;
    FloatGrid firstGood;
    int firstGoodIndex = -1;
    for ( int i = 0; i < grids.size(); ++i )
    {
        if ( grids[i] )
        {
            firstGoodIndex = i;
            firstGood = grids[i];
            break;
        }
    }
    assert( firstGoodIndex >= 0 );
    if ( goodGridsCounter > 1 )
    {
        for ( int i = firstGoodIndex + 1; i < grids.size(); ++i )
        {
            if ( grids[i] )
                firstGood += grids[i];
        }
    }
    size_t numInternalVoxels = 0;
    auto dimsBB = firstGood->evalActiveVoxelBoundingBox();
    auto constAccessor = firstGood->getConstAccessor();
    for ( auto coord : dimsBB )
    {
        if ( constAccessor.isValueOn( coord ) )
        {
            if ( constAccessor.getValue( coord ) < 0.0f )
                ++numInternalVoxels;
        }
    }
    return numInternalVoxels * voxelSize.x * voxelSize.y * voxelSize.z;
}

}