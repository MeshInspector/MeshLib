#include <MRMesh/MRPolylineSubdivide.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRVector2.h>
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, SubdividePolyline)
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 0.f ) );
    Polyline2 polyline( { cont } );

    PolylineSubdivideSettings settings;
    settings.maxEdgeLen = 0.3f;
    settings.maxEdgeSplits = 1000;
    int splitsDone = subdividePolyline( polyline, settings );
    EXPECT_TRUE( splitsDone > 11 && splitsDone < 15 );
}

} //namespace MR
