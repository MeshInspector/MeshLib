#include <MRSymbolMesh/MRAlignTextToMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRSystemPath.h>

namespace MR
{

TEST( MRMesh, BendTextAlongCurve )
{
    const auto sphere = makeUVSphere( 1, 8, 8 );
    sphere.getAABBTree();
    SymbolMeshParams s
    {
        .text = "The quick brown fox jumps over the lazy dog!",
        .pathToFontFile = SystemPath::getFontsDirectory() / "Cousine-Regular.ttf"
    };

    // x = 0.5
    // y = d * sin( p )
    // z = d * cos( p )
    const auto d = sqrt( 3.0f ) / 2;
    auto curve = [d, &sphere]( float p )
    { 
        p *= 6.1f; // slightly less than 2 * PI_F to make a space
        Vector3f pos( 0.5f, d * sin( p ), d * cos( p ) );
        pos = findProjection( pos, sphere ).proj.point;
        return CurvePoint
        {
            .pos = pos,
            .dir = Vector3f( 0, cos( p ), -sin( p ) ),
            .snorm = pos.normalized()
        };
    };

    auto maybeText = bendTextAlongCurve( { s, 0, curve, 0.2f, 0.03f } );
    EXPECT_TRUE( maybeText.has_value() );
    EXPECT_TRUE( maybeText->topology.numValidFaces() > 0 );
}

} //namespace MR
