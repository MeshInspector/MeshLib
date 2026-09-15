#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRUVSphere.h>

#include <iostream>

int main()
{
//! [0]
    // create first sphere with radius of 1 unit
    MR::Mesh sphere1 = MR::makeUVSphere( 1.0f, 64, 64 );

    // create second sphere by cloning the first sphere and moving it in X direction
    MR::Mesh sphere2 = sphere1;
    MR::AffineXf3f xf = MR::AffineXf3f::translation( MR::Vector3f( 0.7f, 0.0f, 0.0f ) );
    sphere2.transform( xf );

    // optional mapper relating the primitives of the input meshes to the primitives of the result
    MR::BooleanResultMapper mapper;

    // perform boolean operation
    MR::BooleanResult result = MR::boolean( sphere1, sphere2, MR::BooleanOperation::Intersection, { .mapper = &mapper } );
    if ( !result.valid() )
        std::cerr << result.errorString << std::endl;

    MR::Mesh resultMesh = *result;
//! [0]

//! [1]
    // find the faces of the result produced by each input sphere, and the faces the cut created
    using MapObject = MR::BooleanResultMapper::MapObject;
    MR::FaceBitSet facesOfSphere1 = mapper.map( sphere1.topology.getValidFaces(), MapObject::A );
    MR::FaceBitSet facesOfSphere2 = mapper.map( sphere2.topology.getValidFaces(), MapObject::B );
    MR::FaceBitSet newFaces = mapper.newFaces();

    std::cout << "faces from sphere1: " << facesOfSphere1.count() << "\n"
              << "faces from sphere2: " << facesOfSphere2.count() << "\n"
              << "faces created by the cut: " << newFaces.count() << std::endl;
//! [1]

//! [2]
    // map one particular face of sphere1 forward: the cut can split it in several faces of the
    // result, or drop it completely if that part of sphere1 is not in the result
    MR::FaceId faceOfSphere1( 793 );
    MR::FaceBitSet oneFace;
    oneFace.autoResizeSet( faceOfSphere1 );
    MR::FaceBitSet producedFaces = mapper.map( oneFace, MapObject::A );
    std::cout << "face " << faceOfSphere1 << " of sphere1 produced " << producedFaces.count() << " faces of the result:";
    for ( MR::FaceId f : producedFaces )
        std::cout << ' ' << f;
    std::cout << std::endl;

    // and backward: the face of sphere1 each face of the result came from
    // (invalid id for the faces that came from sphere2)
    MR::FaceMap new2OldFaces = mapper.getNew2OldFaceMap( MapObject::A );
    MR::FaceId resultFace = producedFaces.find_first();
    std::cout << "face " << resultFace << " of the result came from face " << new2OldFaces[resultFace] << " of sphere1" << std::endl;
//! [2]

    // save result to STL file
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( resultMesh, "out_boolean.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }

    return 0;
}
