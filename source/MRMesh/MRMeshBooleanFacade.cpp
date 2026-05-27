#include "MRMeshBooleanFacade.h"
#include "MRMeshBoolean.h"
#include "MRObjectMesh.h"

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

} //namespace MR
