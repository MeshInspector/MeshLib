#include <MRMesh/MRPolyline2Collide.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MREdgePoint.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, Polyline2Collide )
{
    Vector2f as[2] = { { 0, 1 }, { 4, 5 } };
    Polyline2 a;
    a.addFromPoints( as, 2, false );

    Vector2f bs[2] = { { 0, 2 }, { 2, 0 } };
    Polyline2 b;
    b.addFromPoints( bs, 2, false );

    auto res = findCollidingEdgePairs( a, b );
    ASSERT_EQ( res.size(), 1 );
    ASSERT_EQ( res[0].a.e, 0_e );
    ASSERT_EQ( res[0].a.a, 1.0f / 8 );
    ASSERT_EQ( res[0].b.e, 0_e );
    ASSERT_EQ( res[0].b.a, 1.0f / 4 );
}

TEST( MRMesh, Polyline2SelfCollide )
{
    Vector2f as[2] = { { 0, 1 }, { 4, 5 } };
    Polyline2 polyline;
    polyline.addFromPoints( as, 2, false );

    Vector2f bs[2] = { { 0, 2 }, { 2, 0 } };
    polyline.addFromPoints( bs, 2, false );

    auto res = findSelfCollidingEdgePairs( polyline );
    ASSERT_EQ( res.size(), 1 );
    ASSERT_EQ( res[0].a.e, 2_e );
    ASSERT_EQ( res[0].a.a, 1.0f / 4 );
    ASSERT_EQ( res[0].b.e, 0_e );
    ASSERT_EQ( res[0].b.a, 1.0f / 8 );
}

} //namespace MR
