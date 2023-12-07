#include "MRMeshBooleanFacade.h"
#include "MRMeshBoolean.h"
#include "MRObjectMesh.h"
#include "MRGTest.h"
#include "MRCube.h"
#include "MRMakeSphereMesh.h"

namespace MR
{

TransformedMesh MeshMeshConverter::operator() ( const ObjectMesh & obj ) const
{
    return TransformedMesh( *obj.mesh(), obj.xf() );
}

TransformedMesh & operator += ( TransformedMesh & a, const TransformedMesh& b )
{
    auto b2a = a.xf.inverse() * b.xf;
    auto res = boolean( a.mesh, b.mesh, BooleanOperation::Union, &b2a );
    assert( res.valid() );
    if ( res.valid() )
        a.mesh = std::move( res.mesh );
    return a;
}

TransformedMesh & operator -= ( TransformedMesh & a, const TransformedMesh& b )
{
    auto b2a = a.xf.inverse() * b.xf;
    auto res = boolean( a.mesh, b.mesh, BooleanOperation::DifferenceAB, &b2a );
    assert( res.valid() );
    if ( res.valid() )
        a.mesh = std::move( res.mesh );
    return a;
}

TransformedMesh & operator *= ( TransformedMesh & a, const TransformedMesh& b )
{
    auto b2a = a.xf.inverse() * b.xf;
    auto res = boolean( a.mesh, b.mesh, BooleanOperation::Intersection, &b2a );
    assert( res.valid() );
    if ( res.valid() )
        a.mesh = std::move( res.mesh );
    return a;
}

TEST( MRMesh, MeshBooleanFacade )
{
    Mesh gingivaCopy = makeCube();
    Mesh combinedTooth = makeUVSphere( 1.1f );
    MeshMeshConverter convert;

    auto gingivaGrid = convert( gingivaCopy );
    auto toothGrid = convert( combinedTooth );
    toothGrid -= gingivaGrid;
    auto tooth = std::make_shared<MR::Mesh>( convert( toothGrid ) );
}

} //namespace MR
