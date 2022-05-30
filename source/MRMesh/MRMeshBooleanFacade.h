#pragma once

#include "MRMesh.h"
#include "MRAffineXf3.h"

namespace MR
{

/// just stores a mesh and its transformation to some fixed reference frame
struct TransformedMesh
{
    Mesh mesh;
    AffineXf3f xf;
    TransformedMesh() = default;
    TransformedMesh( Mesh mesh, const AffineXf3f& xf = {} ) : mesh( std::move( mesh ) ), xf( xf ) {}
};

/// the purpose of this class is to be a replacement for MeshVoxelsConverter
/// in case one wants to quickly assess the change from voxel-based to mesh-based boolean
struct MeshMeshConverter
{
    TransformedMesh operator() ( Mesh mesh, const AffineXf3f& xf = {} ) const
        { return TransformedMesh( std::move( mesh ), xf ); }
    MRMESH_API TransformedMesh operator() ( const ObjectMesh & obj ) const;

    const Mesh & operator() ( const TransformedMesh & xm ) const
        { return xm.mesh; }
    Mesh && operator() ( TransformedMesh && xm ) const
        { return std::move( xm.mesh ); }
};

/// union operation on two meshes
MRMESH_API TransformedMesh & operator += ( TransformedMesh & a, const TransformedMesh& b );

/// difference operation on two meshes
MRMESH_API TransformedMesh & operator -= ( TransformedMesh & a, const TransformedMesh& b );

/// intersection operation on two meshes
MRMESH_API TransformedMesh & operator *= ( TransformedMesh & a, const TransformedMesh& b );

} //namespace MR


