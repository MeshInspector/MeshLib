#include <MRSymbolMesh/MRAlignTextToMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, CurvedAlignTextToMesh )
{
    const auto sphere = makeUVSphere( 1, 8, 8 );

    SymbolMeshParams s( "Hello, world!" );

    // x = 0.5
    // y = d * sin( p )
    // z = d * cos( p )
    const auto d = sqrt( 3.0f ) / 2;
    auto curvePos = [d]( float p ) { return Vector3f( 0.5f, d * sin( p ), d * cos( p ) ); };
    auto curveDir = []( float p ) { return Vector3f( 0, cos( p ), -sin( p ) ); };

    auto maybeText = curvedAlignTextToMesh( sphere, { s, 0, curvePos, curveDir, 0.2f, 0.03f } );
    EXPECT_TRUE( maybeText.has_value() );
    EXPECT_TRUE( maybeText->topology.numValidFaces() > 0 );
}

} //namespace MR
