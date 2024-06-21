#include "MRAffineXf.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

#include <cstring>

using namespace MR;

static_assert( sizeof( MRAffineXf3f ) == sizeof( AffineXf3f ) );

MRAffineXf3f mrAffineXf3fNew()
{
    constexpr auto xf = AffineXf3f();
    MRAffineXf3f res;
    std::memcpy( &res, &xf, sizeof( xf ) );
    return res;
}
