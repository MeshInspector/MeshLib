#include "MRMeshOrPoints.h"
#include "MRAffineXf.h"
#include "MRMesh.h"
#include "MRPointCloud.h"

MR_DOTNET_NAMESPACE_BEGIN

MR::MeshOrPointsXf MeshOrPointsXf::ToNative()
{
    auto mesh = dynamic_cast<Mesh^>( obj );
    if ( mesh )
    {
        MR::MeshOrPoints nativeObj( *mesh->getMesh() );
        return MR::MeshOrPointsXf{ .obj = nativeObj, .xf = *xf->xf() };
    }
    
    auto pc = dynamic_cast<PointCloud^>( obj );
    MR::MeshOrPoints nativeObj( *pc->getPointCloud() );
    return MR::MeshOrPointsXf{ .obj = nativeObj, .xf = *xf->xf() };
}

MR_DOTNET_NAMESPACE_END
