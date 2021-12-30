#include "MRMakeRigidXf.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRAligningTransform.h"

namespace MR
{

AffineXf3d makeRigidXf( const MeshPart & mp, const AffineXf3d & meshXf )
{
    PointToPointAligningTransform calc;
    for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
    {
        const Vector3d p{ mp.mesh.triCenter( f ) };
        const double a = mp.mesh.area( f );
        calc.add( p, meshXf( p ), a );
    }
    return calc.calculateTransformationMatrix();
}

AffineXf3f makeRigidXf( const MeshPart & mp, const AffineXf3f & meshXf )
{
    return AffineXf3f{ makeRigidXf( mp, AffineXf3d{ meshXf } ) };
}


} //namespace MR
