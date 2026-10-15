#include "MRVoxels/MRDicom.h"
#ifndef MRVOXELS_NO_DICOM
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRVoxels/MRObjectVoxels.h"
#include "MRViewer/MROpenObjects.h"

#include <gtest/gtest.h>

#include <fstream>

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

namespace
{

// one data element in Explicit VR Little Endian
std::string dicomElement( uint16_t group, uint16_t element, std::string_view vr, std::string value )
{
    if ( value.size() % 2 != 0 )
        value.push_back( vr == "UI" ? '\0' : ' ' );
    std::string res;
    auto put = [&res] ( auto v ) { res.append( reinterpret_cast<const char*>( &v ), sizeof( v ) ); };
    put( group );
    put( element );
    res += vr;
    if ( vr == "OB" || vr == "OW" )
    {
        put( uint16_t( 0 ) );
        put( uint32_t( value.size() ) );
    }
    else
        put( uint16_t( value.size() ) );
    return res + value;
}

template <typename T>
std::string dicomBinary( const T& v )
{
    return std::string( reinterpret_cast<const char*>( &v ), sizeof( T ) );
}

// single-frame 16-bit slice of the given SOP class with spacing, position and rescale in the top-level elements
void writeDicomSlice( const std::filesystem::path& path, const std::string& sopClass, int instance, double z,
    int cols, int rows, const std::vector<uint16_t>& pixels, const std::string& photometric = "MONOCHROME2" )
{
    const auto sopInstance = "1.2.3.4.5." + std::to_string( instance );
    const auto meta =
        dicomElement( 0x0002, 0x0001, "OB", std::string( "\0\1", 2 ) ) +
        dicomElement( 0x0002, 0x0002, "UI", sopClass ) +
        dicomElement( 0x0002, 0x0003, "UI", sopInstance ) +
        dicomElement( 0x0002, 0x0010, "UI", "1.2.840.10008.1.2.1" ); // Explicit VR Little Endian
    const auto dataSet =
        dicomElement( 0x0008, 0x0016, "UI", sopClass ) +
        dicomElement( 0x0008, 0x0018, "UI", sopInstance ) +
        dicomElement( 0x0008, 0x0060, "CS", "CT" ) +
        dicomElement( 0x0020, 0x000E, "UI", "1.2.3.4.5" ) + // SeriesInstanceUID
        dicomElement( 0x0020, 0x0013, "IS", std::to_string( instance ) ) +
        dicomElement( 0x0020, 0x0032, "DS", "-10\\-20\\" + std::to_string( z ) ) + // ImagePositionPatient
        dicomElement( 0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0" ) + // ImageOrientationPatient
        dicomElement( 0x0028, 0x0002, "US", dicomBinary( uint16_t( 1 ) ) ) + // SamplesPerPixel
        dicomElement( 0x0028, 0x0004, "CS", photometric ) +
        dicomElement( 0x0028, 0x0010, "US", dicomBinary( uint16_t( rows ) ) ) +
        dicomElement( 0x0028, 0x0011, "US", dicomBinary( uint16_t( cols ) ) ) +
        dicomElement( 0x0028, 0x0030, "DS", "0.5\\0.5" ) + // PixelSpacing
        dicomElement( 0x0028, 0x0100, "US", dicomBinary( uint16_t( 16 ) ) ) + // BitsAllocated
        dicomElement( 0x0028, 0x0101, "US", dicomBinary( uint16_t( 16 ) ) ) + // BitsStored
        dicomElement( 0x0028, 0x0102, "US", dicomBinary( uint16_t( 15 ) ) ) + // HighBit
        dicomElement( 0x0028, 0x0103, "US", dicomBinary( uint16_t( 0 ) ) ) + // PixelRepresentation
        dicomElement( 0x0028, 0x1052, "DS", "-1000" ) + // RescaleIntercept
        dicomElement( 0x0028, 0x1053, "DS", "2" ) + // RescaleSlope
        dicomElement( 0x7FE0, 0x0010, "OW", std::string( reinterpret_cast<const char*>( pixels.data() ), pixels.size() * sizeof( uint16_t ) ) );

    std::ofstream out( path, std::ios::binary );
    out << std::string( 128, '\0' ) << "DICM"
        << dicomElement( 0x0002, 0x0000, "UL", dicomBinary( uint32_t( meta.size() ) ) ) << meta << dataSet;
}

} // anonymous namespace

// OnDemand3D exports CBCT as single-frame slices tagged Enhanced CT Image Storage without functional groups,
// they must load as the same slices tagged CT Image Storage
TEST( MRMesh, DicomLoadSlicesWithoutFunctionalGroups )
{
    constexpr int cols = 3;
    constexpr int rows = 2;
    constexpr int numSlices = 3;
    for ( const std::string sopClass : { "1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1" } )
    {
        SCOPED_TRACE( sopClass );
        UniqueTemporaryFolder tmpFolder;
        // instance numbers ascend while z descends
        for ( int i = 0; i < numSlices; ++i )
        {
            std::vector<uint16_t> pixels( cols * rows );
            for ( int p = 0; p < cols * rows; ++p )
                pixels[p] = uint16_t( 500 + 100 * i + p );
            writeDicomSlice( tmpFolder / ( "slice" + std::to_string( i ) + ".dcm" ), sopClass, i + 1, 0.75 * ( numSlices - 1 - i ),
                cols, rows, pixels );
        }

        auto loadRes = VoxelsLoad::loadDicomsFolder( tmpFolder, 4 );
        ASSERT_EQ( loadRes.size(), 1u );
        ASSERT_TRUE( loadRes[0].has_value() ) << loadRes[0].error();
        const auto& dcm = *loadRes[0];
        const SimpleVolumeMinMax& vol = dcm.vol;
        EXPECT_EQ( vol.dims, Vector3i( cols, rows, numSlices ) );
        EXPECT_NEAR( vol.voxelSize.x, 0.5e-3f, 1e-9f );
        EXPECT_NEAR( vol.voxelSize.y, 0.5e-3f, 1e-9f );
        EXPECT_NEAR( vol.voxelSize.z, 0.75e-3f, 1e-9f );
        // the volume starts at the lowest slice
        EXPECT_NEAR( dcm.xf.b.z, 0.f, 1e-6f );
        ASSERT_EQ( vol.data.size(), size_t( cols * rows * numSlices ) );
        for ( int z = 0; z < numSlices; ++z )
        {
            const int i = numSlices - 1 - z;
            for ( int p = 0; p < cols * rows; ++p )
                EXPECT_EQ( vol.data[VoxelId( size_t( z * cols * rows + p ) )], 2.f * float( 500 + 100 * i + p ) - 1000.f ) << "z=" << z << " p=" << p;
        }
    }
}

// an unsupported file (e.g. a color thumbnail) listed first must not block opening the folder
TEST( MRMesh, DicomFolderWithUnsupportedFile )
{
    UniqueTemporaryFolder tmpFolder;
    const std::string ct = "1.2.840.10008.5.1.4.1.1.2";
    const std::vector<uint16_t> pixels = { 500, 1500, 2500, 3500 };
    writeDicomSlice( tmpFolder / "a.dcm", ct, 1, 0, 2, 2, pixels, "YBR_FULL" );
    for ( int i = 0; i < 3; ++i )
        writeDicomSlice( tmpFolder / ( "slice" + std::to_string( i ) + ".dcm" ), ct, i + 1, i, 2, 2, pixels );

    auto res = makeObjectTreeFromFolder( tmpFolder, true, [] ( float ) { return true; } );
    ASSERT_TRUE( res.has_value() ) << res.error();
    auto voxels = std::dynamic_pointer_cast<ObjectVoxels>( res->obj );
    ASSERT_TRUE( voxels );
    EXPECT_EQ( voxels->vdbVolume().dims, Vector3i( 2, 2, 3 ) );
}

} //namespace MR

#endif //!MRVOXELS_NO_DICOM
