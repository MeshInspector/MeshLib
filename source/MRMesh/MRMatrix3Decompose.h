#pragma once

#include "MRMesh/MRToFromEigen.h"
#include <Eigen/Dense>

namespace MR
{

/// Decomposes affine transformation matrix into rotation and scaling matrices
template <typename T>
void decomposeMatrix3( const Matrix3<T>& m, Matrix3<T>& rotation, Matrix3<T>& scaling )
{
    Eigen::HouseholderQR<Eigen::MatrixXf> qr( toEigen( m ) );
    auto q = fromEigen( Eigen::Matrix3f{ qr.householderQ() } );
    auto r = fromEigen( Eigen::Matrix3f{ qr.matrixQR() } );

    scaling = {};
    Matrix3<T> sign;
    for ( int i = 0; i < 3; ++i )
    {
        scaling[i][i] = std::abs( r[i][i] );
        if ( r[i][i] < 0 )
            sign[i][i] = -1;
    }
    rotation = q * sign;
}

} // namespace MR