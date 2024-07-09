#include "MRMeshOrPoints.h"

#include "MRMesh/MRMeshOrPoints.h"

using namespace MR;

MRMeshOrPoints* mrMeshOrPointsFromMesh( const MRMesh* mesh_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );

    return reinterpret_cast<MRMeshOrPoints*>( new MeshOrPoints( mesh ) );
}

MRMeshOrPoints* mrMeshOrPointsFromPointCloud( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<MRMeshOrPoints*>( new MeshOrPoints( pc ) );
}

MRMeshOrPointsXf* mrMeshOrPointsXfNew( const MRMeshOrPoints* obj_, const MRAffineXf3f* xf_ )
{
    const auto& obj = *reinterpret_cast<const MeshOrPoints*>( obj_ );
    const auto& xf = *reinterpret_cast<const AffineXf3f*>( xf_ );

    return reinterpret_cast<MRMeshOrPointsXf*>( new MeshOrPointsXf( obj, xf ) );
}

MRMeshOrPointsXf* mrMeshOrPointsXfFromMesh( const MRMesh* mesh_, const MRAffineXf3f* xf_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    const auto& xf = *reinterpret_cast<const AffineXf3f*>( xf_ );

    return reinterpret_cast<MRMeshOrPointsXf*>( new MeshOrPointsXf( mesh, xf ) );
}

MRMeshOrPointsXf* mrMeshOrPointsXfFromPointCloud( const MRPointCloud* pc_, const MRAffineXf3f* xf_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );
    const auto& xf = *reinterpret_cast<const AffineXf3f*>( xf_ );

    return reinterpret_cast<MRMeshOrPointsXf*>( new MeshOrPointsXf( pc, xf ) );
}
