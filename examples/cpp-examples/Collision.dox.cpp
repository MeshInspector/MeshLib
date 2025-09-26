#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMeshCollide.h"
#include <iostream>

int main()
{
    auto meshA = MR::makeUVSphere(); // make mesh A
    auto meshB = MR::makeUVSphere(); // make mesh B
    meshB.transform( MR::AffineXf3f::translation( MR::Vector3f( 0.1f, 0.1f, 0.1f ) ) ); // shift mesh B for better demonstration

    auto collidingFacePairs = MR::findCollidingTriangles( meshA, meshB ); // find each pair of colliding faces
    for ( const auto [fa, fb] : collidingFacePairs )
        std::cout << int( fa ) << " " << int( fb ) << "\n"; // print each pair of colliding faces

    auto [collidingFaceBitSetA, collidingFaceBitSetB] = MR::findCollidingTriangleBitsets( meshA, meshB ); // find bitsets of colliding faces
    std::cout << collidingFaceBitSetA.count() << "\n"; // print number of colliding faces from mesh A
    std::cout << collidingFaceBitSetB.count() << "\n"; // print number of colliding faces from mesh B


    auto isColliding = !MR::findCollidingTriangles( meshA, meshB, nullptr, true ).empty(); // fast check if mesh A and mesh B collide
    std::cout << isColliding << "\n";
    return 0;
}
