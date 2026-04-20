#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRMesh/MRZip.h>
#include <MRPch/MRSpdlog.h>

#include <filesystem>

namespace MR
{

// Writes a ~100K-vertex sphere to a .mrmesh file in a temporary folder, then
// compresses that folder to a .zip and verifies the archive was created and
// is non-empty. Serves as a realistic end-to-end exercise of MeshLib's zip
// write path (libzip + deflate) on mesh-sized data.
TEST( MRMesh, CompressSphereToZip )
{
    // No-op re-trigger marker for CI; remove in cleanup commit.
    UniqueTemporaryFolder srcFolder;
    ASSERT_TRUE( bool( srcFolder ) );

    // Generate a sphere with ~100K vertices. makeSphere's subdivision
    // targets the requested count but may land a handful over.
    constexpr int targetVerts = 100'000;
    SphereParams params;
    params.radius = 1.0f;
    params.numMeshVertices = targetVerts;
    const Mesh sphere = makeSphere( params );
    EXPECT_EQ( (int)sphere.topology.numValidVerts(), targetVerts );

    // Save mesh as a .mrmesh file in the temp folder.
    const std::filesystem::path meshPath = srcFolder / "sphere.mrmesh";
    const auto saveRes = MeshSave::toMrmesh( sphere, meshPath );
    ASSERT_TRUE( saveRes.has_value() ) << saveRes.error();
    std::error_code ec;
    ASSERT_TRUE( std::filesystem::exists( meshPath, ec ) );
    const auto meshSize = std::filesystem::file_size( meshPath, ec );
    EXPECT_GT( meshSize, 0u );
    spdlog::info( "sphere.mrmesh size: {} bytes", meshSize );

    // Compress the temp folder into a .zip located in a second temp folder
    // (so the zip isn't inside the folder being compressed).
    UniqueTemporaryFolder dstFolder;
    ASSERT_TRUE( bool( dstFolder ) );
    const std::filesystem::path zipPath = dstFolder / "sphere.zip";

    const auto compressRes = compressZip( zipPath, srcFolder );
    ASSERT_TRUE( compressRes.has_value() ) << compressRes.error();
    ASSERT_TRUE( std::filesystem::exists( zipPath, ec ) );
    const auto zipSize = std::filesystem::file_size( zipPath, ec );
    EXPECT_GT( zipSize, 0u );
    spdlog::info( "sphere.zip size:    {} bytes", zipSize );

    // Sanity: the zip should not be absurdly larger than the source
    // (that would indicate something is wrong with the envelope); and
    // since .mrmesh is a raw binary dump of topology plus coordinate
    // floats, deflate typically produces a modestly smaller archive.
    EXPECT_LT( zipSize, meshSize * 2u );
}

} // namespace MR
