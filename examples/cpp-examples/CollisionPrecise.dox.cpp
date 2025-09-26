#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include <iostream>

int main()
{
    auto meshA = MR::makeUVSphere(); // make mesh A
    auto meshB = MR::makeUVSphere(); // make mesh B
    meshB.transform( MR::AffineXf3f::translation( MR::Vector3f( 0.1f, 0.1f, 0.1f ) ) ); // shift mesh B for better demonstration

    auto converters = MR::getVectorConverters( meshA, meshB ); // create converters to integer field (needed for absolute precision predicates)
    auto collidingFaceEdges = MR::findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt ); // find each intersecting edge/triangle pair 
    // print pairs of edges triangles
    for ( const auto& vet : collidingFaceEdges )
    {
        if ( vet.isEdgeATriB() )
            std::cout << "edgeA: " << int( vet.edge ) << ", triB: " << vet.tri().get() << "\n";
        else
            std::cout << "triA: " << vet.tri().get() << ", edgeB: " << int( vet.edge ) << "\n";
    }
    return 0;
}
