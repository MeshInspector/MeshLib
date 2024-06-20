#include "MRAffineXf.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

using namespace MR;

static_assert( sizeof( MRAffineXf3f ) == sizeof( AffineXf3f ) );

MRAffineXf3f mrAffineXf3fNew()
{
    constexpr auto xf = AffineXf3f();
    return *reinterpret_cast<const MRAffineXf3f*>( &xf );
}
