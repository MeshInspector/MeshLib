#include "MRVoxels/MRVoxelsLoad.h"
#ifndef MRVOXELS_NO_TIFF
#include "MRVoxels/MRMarchingCubes.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTiffIO.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"

#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, LoadTiffDir )
{
    const Vector3i dims{ 10, 8, 4 };
    UniqueTemporaryFolder tmpFolder;
    std::vector<uint8_t> slice( size_t( dims.x ) * dims.y );
    for ( int z = 0; z < dims.z; ++z )
    {
        // box of 3x2x2 voxels with value 255 in slices 1 and 2, all other voxels are zeros
        std::fill( slice.begin(), slice.end(), uint8_t( 0 ) );
        if ( z == 1 || z == 2 )
            for ( int y = 1; y <= 2; ++y )
                for ( int x = 2; x <= 4; ++x )
                    slice[y * dims.x + x] = 255;
        auto saveRes = writeRawTiff( slice.data(), tmpFolder / ( "slice" + std::to_string( z ) + ".tif" ), { .baseParams = {
            .sampleType = BaseTiffParameters::SampleType::Uint,
            .valueType = BaseTiffParameters::ValueType::Scalar,
            .bytesPerSample = 1,
            .imageSize = { dims.x, dims.y }
        } } );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    }

    auto vol = VoxelsLoad::loadTiffDir( { .dir = tmpFolder } );
    ASSERT_TRUE( vol.has_value() ) << vol.error();
    EXPECT_EQ( vol->dims, dims );
    // zero voxels of every slice must be active, otherwise marchingCubes considers them invalid
    EXPECT_EQ( vol->data->activeVoxelCount(), size_t( dims.x ) * dims.y * dims.z );

    MarchingCubesParams params;
    params.iso = 127.5f;
    auto mesh = marchingCubes( *vol, params );
    ASSERT_TRUE( mesh.has_value() ) << mesh.error();
    EXPECT_TRUE( mesh->topology.isClosed() );
    const auto box = mesh->computeBoundingBox();
    EXPECT_NEAR( ( box.min - Vector3f( 1.5f, 0.5f, 0.5f ) ).length(), 0.f, 1e-6f );
    EXPECT_NEAR( ( box.max - Vector3f( 4.5f, 2.5f, 2.5f ) ).length(), 0.f, 1e-6f );
}

} //namespace MR

#endif //!MRVOXELS_NO_TIFF
