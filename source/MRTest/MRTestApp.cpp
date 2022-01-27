#include <MRMesh/MRMesh.h>
#include "MRMesh/MRLog.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRQuadraticForm.h"

namespace MR
{

TEST(MeshRUs, QuadraticForm)
{
    QuadraticForm3f q0, q1;
    q0.addDistToOrigin( 1 );
    q1.addDistToOrigin( 1 );
    auto r = sum( q0, Vector3f{0,0,0}, q1, Vector3f{2,0,0} );

    EXPECT_EQ( r.second, (Vector3f{1,0,0}) );
}

} //namespace MR

int main(int argc, char **argv)
{
    MR::loadMeshDll();
    MR::setupLoggerByDefault();

    // print compiler info
#ifdef __clang__
    spdlog::info( "{}", __VERSION__ );
#elif defined __GNUC__
    spdlog::info( "GCC {}", __VERSION__ );
#else
    spdlog::info( "MSVC {}", _MSC_FULL_VER );
#endif

    std::vector<std::string> xs{"text0", "text1"};
    fmt::format( "Test {}", fmt::join( xs, "," ) );

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
