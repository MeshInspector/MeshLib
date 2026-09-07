#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMeshCollide.h"
#include <iostream>

int main()
{
    auto mesh = MR::makeTorusWithSelfIntersections(); // make torus with self-intersections

    auto selfCollidingParis = MR::findSelfCollidingTriangles( mesh ); // find self-intersecting faces pairs
    if ( !selfCollidingParis.has_value() )
    {
        // check error
        std::cerr << selfCollidingParis.error();
        return 1;
    }
    for ( auto [fl, fr] : *selfCollidingParis )
        std::cout << int( fl ) << " " << int( fr ) << "\n"; // print each pair

    auto selfCollidingBitSet = MR::findSelfCollidingTrianglesBS( mesh ); // find bitset of self-intersecting faces
    if ( !selfCollidingBitSet.has_value() )
    {
        // check error
        std::cerr << selfCollidingBitSet.error();
        return 1;
    }
    std::cout << selfCollidingBitSet->count() << "\n"; // print number of self-intersecting faces

    auto isSelfColliding = MR::findSelfCollidingTriangles( mesh, nullptr ); // fast check if mesh has self-intersections
    if ( !isSelfColliding.has_value() )
    {
        // check error
        std::cerr << isSelfColliding.error();
        return 1;
    }
    std::cout << *isSelfColliding << "\n";
    return 0;
}
