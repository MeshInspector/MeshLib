#ifndef MRVOXELS_NO_DICOM

#include "MRVoxels/MRDicom.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRStreamOperators.h"
#include "MRMesh/MRGTest.h"

namespace MR
{

TEST( MRMesh, DicomSaveLoad )
{
    SimpleVolumeU16 a
    {
        .data = { 1, 2, 3, 4, 5, 6, 7, 8 },
        .dims = { 2, 2, 2 },
        .voxelSize = { 1.f, 2.f, 3.f }
    };
    // range of uint16 to load float values as is
    const std::optional<MinMaxf> sourceScale = MinMaxf{ 0, 65535 };

    UniqueTemporaryFolder tmpFolder;
    auto filename = tmpFolder / "a.dcm";
    auto saveRes = VoxelsSave::toDicom( a, filename, sourceScale );
    EXPECT_TRUE( saveRes.has_value() );

    auto loadRes = VoxelsLoad::loadDicomFile( filename );
    EXPECT_TRUE( loadRes.has_value() );

    const SimpleVolumeMinMax& b = loadRes->vol;
    EXPECT_EQ( a.dims, b.dims );
    EXPECT_NEAR( a.voxelSize.x, b.voxelSize.x, 1e-6f );
    EXPECT_NEAR( a.voxelSize.y, b.voxelSize.y, 1e-6f );
    EXPECT_NEAR( a.voxelSize.z, b.voxelSize.z, 1e-6f );
    EXPECT_EQ( a.data.size(), b.data.size() );
    for ( VoxelId i = 0_vox; i < a.data.size(); ++i )
        EXPECT_EQ( a.data[i], b.data[i] );
}

} //namespace MR

#endif //!MRVOXELS_NO_DICOM
