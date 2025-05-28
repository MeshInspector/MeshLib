#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSetParallelFor.h>
#include <MRMesh/MRGTest.h>
#include <MRVoxels/MRMarchingCubes.h>

namespace MR
{

TEST( MRMesh, MarchingCubes )
{
    SimpleVolume fltVol
    {
        .dims = { 10, 10, 10 },
        .voxelSize = { 1, 1, 1 }
    };
    fltVol.data.reserve( 1000 );
    for ( int z = 0; z < 10; ++z )
    {
        for ( int y = 0; y < 10; ++y )
        {
            for ( int x = 0; x < 10; ++x )
            {
                fltVol.data.push_back( Vector3f( x - 4.5f, y - 4.5f, z - 4.5f ).lengthSq() < sqr( 4 ) );
            }
        }
    }

    MarchingCubesParams vparams;
    vparams.iso = 0.5f;
    vparams.cb = [](float) { return true; };
    auto maybeMeshA = marchingCubes( fltVol, vparams );
    EXPECT_TRUE( maybeMeshA.has_value() );
    EXPECT_TRUE( maybeMeshA->topology.numValidFaces() > 0 );

    SimpleBinaryVolume binVol
    {
        .data = VoxelBitSet{ fltVol.data.size() },
        .dims = fltVol.dims,
        .voxelSize = fltVol.voxelSize
    };
    BitSetParallelForAll( binVol.data, [&]( VoxelId v )
    {
        binVol.data[v] = fltVol.data[v] > 0.5f;
    } );
    auto maybeMeshB = marchingCubes( binVol, vparams );
    EXPECT_TRUE( maybeMeshB.has_value() );
    EXPECT_TRUE( maybeMeshB->topology.numValidFaces() > 0 );
    EXPECT_EQ( *maybeMeshA, *maybeMeshB );
}

} //namespace MR
