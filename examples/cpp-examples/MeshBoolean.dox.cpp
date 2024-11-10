#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRUVSphere.h>

#include <iostream>

int main()
{
    // create first sphere with radius of 1 unit
    MR::Mesh sphere1 = MR::makeUVSphere( 1.0f, 64, 64 );

    // create second sphere by cloning the first sphere and moving it in X direction
    MR::Mesh sphere2 = sphere1;
    MR::AffineXf3f xf = MR::AffineXf3f::translation( MR::Vector3f( 0.7f, 0.0f, 0.0f ) );
    sphere2.transform( xf );

    // perform boolean operation
    MR::BooleanResult result = MR::boolean( sphere1, sphere2, MR::BooleanOperation::Intersection );
    MR::Mesh resultMesh = *result;
    if ( !result.valid() )
        std::cerr << result.errorString << std::endl;

    // save result to STL file
    MR::MeshSave::toAnySupportedFormat( resultMesh, "out_boolean.stl" );

    return 0;
}
