#include "MRMesh/MRMeshDelone.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRGTest.h"

namespace MR
{

TEST( MRMesh, checkDeloneQuadrangle )
{
    Vector3d a( 0, 0, 0 );
    Vector3d b( 1, 0, 0 );
    Vector3d c( 2, 0, 0 );
    Vector3d d( 1.1, 0, 0 );
    // ABCD quadrangle has zero area
    EXPECT_FALSE( checkDeloneQuadrangle( a, b, c, d, 10 ) );
}

} //namespace MR
