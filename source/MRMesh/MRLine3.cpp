#include "MRLine3.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, Line3)
{
    {
        MR::Vector3f p( float(-393.72412), float(-373.53027), float(230.45459) );
        MR::Vector3f d( float(9461.5488), float(9197.1641), float(-6009.543) );
        MR::Vector3f pnt( float(-23.058762), float(-13.051398), float(-5.1603823) );
        auto d2 = MR::Line3f(p, d).distanceSq(pnt);
        EXPECT_GE( d2, 0.02925f );
        EXPECT_LE( d2, 0.02926f );
    }

    {
        MR::Vector3d p( 0, 0, 0 );
        MR::Vector3d d( 1, 0, 0 );
        MR::Vector3d pnt( 1e8, 0, 0.1 );
        auto d2 = MR::Line3d(p, d).distanceSq(pnt);
        EXPECT_GE( d2, 0.00999 );
        EXPECT_LE( d2, 0.01001 );
    }
}

} //namespace MR
