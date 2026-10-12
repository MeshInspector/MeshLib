#include "MRIOExtras/MRTiff.h"
#ifndef MRIOEXTRAS_NO_TIFF
#include "MRVoxels/MRMarchingCubes.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRVoxels/MRVoxelsLoad.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"

#include <gtest/gtest.h>

#include <fstream>

namespace MR
{

TEST( MRMesh, LoadRasterFolder )
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

    auto vol = VoxelsLoad::loadRasterFolder( { .dir = tmpFolder } );
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

TEST( MRMesh, LoadRasterFolderNegativeValues )
{
    UniqueTemporaryFolder tmpFolder;
    for ( int z = 0; z < 2; ++z )
    {
        const std::vector<float> slice( 4, -1.f - z );
        auto saveRes = writeRawTiff( ( const uint8_t* )slice.data(), tmpFolder / ( "slice" + std::to_string( z ) + ".tif" ), { .baseParams = {
            .sampleType = BaseTiffParameters::SampleType::Float,
            .valueType = BaseTiffParameters::ValueType::Scalar,
            .bytesPerSample = 4,
            .imageSize = { 2, 2 }
        } } );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    }

    auto vol = VoxelsLoad::loadRasterFolder( { .dir = tmpFolder } );
    ASSERT_TRUE( vol.has_value() ) << vol.error();
    EXPECT_EQ( vol->min, -2.f );
    EXPECT_EQ( vol->max, -1.f );
}

// the slices are the files of the extension with a raster loader that most files have, and they must have the same size and format
TEST( MRMesh, LoadRasterFolderFiles )
{
    UniqueTemporaryFolder tmpFolder;
    auto writeSlice = [&] ( const std::string& name, const Vector2i& size )
    {
        const std::vector<uint8_t> slice( size_t( size.x ) * size.y, 7 );
        auto saveRes = writeRawTiff( slice.data(), tmpFolder / name, { .baseParams = {
            .sampleType = BaseTiffParameters::SampleType::Uint,
            .valueType = BaseTiffParameters::ValueType::Scalar,
            .bytesPerSample = 1,
            .imageSize = size
        } } );
        ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    };
    for ( int z = 0; z < 3; ++z )
        writeSlice( "slice" + std::to_string( z ) + ".tif", { 2, 3 } );
    // fewer files of another raster extension, and more files without a raster loader
    writeSlice( "other0.tiff", { 4, 4 } );
    writeSlice( "other1.tiff", { 4, 4 } );
    for ( int i = 0; i < 4; ++i )
        std::ofstream( tmpFolder / ( "note" + std::to_string( i ) + ".txt" ) ) << "not a raster";

    auto vol = VoxelsLoad::loadRasterFolder( { .dir = tmpFolder } );
    ASSERT_TRUE( vol.has_value() ) << vol.error();
    EXPECT_EQ( vol->dims, Vector3i( 2, 3, 3 ) );

    writeSlice( "slice3.tif", { 3, 2 } );
    vol = VoxelsLoad::loadRasterFolder( { .dir = tmpFolder } );
    ASSERT_FALSE( vol.has_value() );
    EXPECT_NE( vol.error().find( "slice3.tif" ), std::string::npos ) << vol.error();
}

} //namespace MR

#endif //!MRIOEXTRAS_NO_TIFF
