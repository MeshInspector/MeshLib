#pragma once
#include "exports.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf3.h"

namespace MRE
{

/// just stores a mesh and its transformation to some fixed reference frame
struct TransformedMesh
{
    MR::Mesh mesh;
    MR::AffineXf3f xf;
    TransformedMesh() = default;
    TransformedMesh( MR::Mesh mesh, const MR::AffineXf3f& xf = {} ) : mesh( std::move( mesh ) ), xf( xf ) {}
};

/// the purpose of this class is to be a replacement for MeshVoxelsConverter
/// in case one wants to quickly assess the change from voxel-based to mesh-based boolean
struct MeshMeshConverter
{
    TransformedMesh operator() ( MR::Mesh mesh, const MR::AffineXf3f& xf = {} ) const
        { return TransformedMesh( std::move( mesh ), xf ); }
    MREALGORITHMS_API TransformedMesh operator() ( const MR::ObjectMesh & obj ) const;

    const MR::Mesh & operator() ( const TransformedMesh & xm ) const
        { return xm.mesh; }
    MR::Mesh && operator() ( TransformedMesh && xm ) const
        { return std::move( xm.mesh ); }
};

/// union operation on two meshes
MREALGORITHMS_API TransformedMesh & operator += ( TransformedMesh & a, const TransformedMesh& b );

/// difference operation on two meshes
MREALGORITHMS_API TransformedMesh & operator -= ( TransformedMesh & a, const TransformedMesh& b );

/// intersection operation on two meshes
MREALGORITHMS_API TransformedMesh & operator *= ( TransformedMesh & a, const TransformedMesh& b );

} //namespace MRE
