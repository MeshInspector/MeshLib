#pragma once

#include "MRMatrix3.h"
#include "MRAffineXf.h"

namespace MR
{

/**
 * \brief computes rigid transformation xf
 * \details xf.z - directed from center to eye \n
 * xf.x - directed orthogonal to up and xf.z \n
 * xf.y - directed orthogonal to xf.z and xf.x \n
 * xf(eye) = 0
 * \ingroup MathGroup
 */
template<typename T>
AffineXf3<T> lookAt( const Vector3<T>& center, const Vector3<T>& eye, const Vector3<T>& up )
{
    const auto f = (center - eye).normalized();
    const auto s = cross( f, up ).normalized();
    const auto u = cross( s, f );

    return AffineXf3f{ 
        Matrix3f{ s, u, -f },
        Vector3f{ -dot( s, eye ), -dot( u, eye ), dot( f, eye ) }
    };
}

} // namespace MR
