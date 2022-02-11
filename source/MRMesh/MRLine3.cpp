#include "MRLine3.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, Line3)
{
    {
        MR::Vector3d p( -393.72412, -373.53027, 230.45459 );
        MR::Vector3d d( 9461.5488, 9197.1641, -6009.543 );
        MR::Vector3d pnt( -23.058762, -13.051398, -5.1603823 );
        EXPECT_GE( MR::Line3d(p, d).distanceSq(pnt), 0.02925 );
        EXPECT_LE( MR::Line3d(p, d).distanceSq(pnt), 0.02926 );
    }

    {
        MR::Vector3f p( float(-393.72412), float(-373.53027), float(230.45459) );
        MR::Vector3f d( float(9461.5488), float(9197.1641), float(-6009.543) );
        MR::Vector3f pnt( float(-23.058762), float(-13.051398), float(-5.1603823) );
        EXPECT_GE( MR::Line3f(p, d).distanceSq(pnt), 0 );
    }
}

} //namespace MR
