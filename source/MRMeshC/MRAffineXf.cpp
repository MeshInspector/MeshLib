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

MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b_ )
{
    const auto& b = *reinterpret_cast<const Vector3f*>( b_ );

    const auto res = AffineXf3f::translation( b );
    const auto* ptr = reinterpret_cast<const MRAffineXf3f*>( &res );
    return { *ptr };
}

MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A_ )
{
    const auto& A = *reinterpret_cast<const Matrix3f*>( A_ );

    const auto res = AffineXf3f::linear( A );
    return *reinterpret_cast<const MRAffineXf3f*>( &res );
}

MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a_, const MRAffineXf3f* b_ )
{
    const auto& a = *reinterpret_cast<const AffineXf3f*>( a_ );
    const auto& b = *reinterpret_cast<const AffineXf3f*>( b_ );

    const auto res = a * b;
    return *reinterpret_cast<const MRAffineXf3f*>( &res );
}
