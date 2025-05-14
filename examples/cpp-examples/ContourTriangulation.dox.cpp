#include <MRMesh/MRImageLoad.h>
#include <MRMesh/MRDistanceMap.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MR2DContoursTriangulation.h>
#include <MRMesh/MRMeshSave.h>
#include <iostream>

int main()
{
    // load image as distance map
    auto imageLoadRes = MR::ImageLoad::fromAnySupportedFormat( "logo.jpg" );
    if ( !imageLoadRes.has_value() )
    {
        std::cerr << imageLoadRes.error() << "\n";
        return 1; // error while loading file
    }
    auto dmRes = MR::convertImageToDistanceMap( *imageLoadRes );
    if ( !dmRes.has_value() )
    {
        std::cerr << dmRes.error() << "\n";
        return 1; // error while converting image to distance map
    }

    auto pl2 = MR::distanceMapTo2DIsoPolyline( *dmRes, 210.0f );
    auto triangulationRes = MR::PlanarTriangulation::triangulateContours( pl2.contours() );

    auto saveRes = MR::MeshSave::toAnySupportedFormat( triangulationRes, "LogoMesh.ctms" );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error() << "\n";
        return 1; // error while saving file
    }
    return 0;
}
