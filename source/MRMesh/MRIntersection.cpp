#include "MRIntersection.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, Intersect) 
{
    Plane3d plane( Vector3d( 1, 0, 0 ), 1 );
    Line3d line( Vector3d( 0, 0, 0 ), Vector3d( 1, 1, 1 ).normalized() );
    auto is0 = intersection( plane, line );
    ASSERT_NEAR( ( *is0 - Vector3d{ 1., 1., 1. } ).length(), 0., 1e-15 );

    Line3d line1( Vector3d( 0, 0, 0 ), Vector3d( 0, 1, 0 ) );
    auto is1 = intersection( plane, line1 );
    ASSERT_FALSE( is1.has_value() );
}

} //namespace MR
