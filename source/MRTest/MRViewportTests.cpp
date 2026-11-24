#include <MRViewer/MRViewport.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRConstants.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRViewer, FindCameraLookingFaces )
{
    // the camera looks at the cube from its (+X,+Y,+Z) corner, and they are located on different sides of the world origin
    const Vector3f toCamera( 1, 1, 1 );
    const Vector3f center( -300, 0, 0 );
    Viewport viewport;
    viewport.cameraLookAlong( -toCamera, Vector3f( -1, -1, 2 ) );
    viewport.setCameraPoint( center + 100.f * toCamera );

    const Mesh cube = makeCube();
    for ( const auto& meshToWorld : {
        AffineXf3f::translation( center ),
        AffineXf3f::translation( center ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusX(), PI2_F ) * Matrix3f::scale( 1.f, 2.f, 3.f ) ) } )
    {
        // three sides of the cube are turned to the camera
        FaceBitSet expected( cube.topology.faceSize() );
        for ( auto f : cube.topology.getValidFaces() )
            if ( dot( meshToWorld.A * cube.normal( f ), toCamera ) > 0 )
                expected.set( f );
        EXPECT_EQ( expected.count(), 6u );

        for ( bool orthographic : { true, false } )
        {
            auto params = viewport.getParameters();
            params.orthographic = orthographic;
            viewport.setParameters( params );
            viewport.setupView();
            EXPECT_EQ( viewport.findCameraLookingFaces( cube, meshToWorld ), expected );
        }
    }
}

} //namespace MR
