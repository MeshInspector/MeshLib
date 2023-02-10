#pragma once

#include "MRMesh/MRToFromEigen.h"
#include <Eigen/Dense>

namespace MR
{

/// Decomposes matrix into rotation and scaling matrices
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

/// Returns true if matrix scale is identity
template <typename T>
bool isRigid( const Matrix3<T>& m )
{
    Matrix3<T> rot, scale;
    decomposeMatrix3( m, rot, scale );
    auto eps = T( 10 ) * std::numeric_limits<T>::epsilon();
    if ( std::abs( scale.x.x - T( 1 ) ) > eps )
        return false;
    if ( std::abs( scale.y.y - T( 1 ) ) > eps )
        return false;
    if ( std::abs( scale.z.z - T( 1 ) ) > eps )
        return false;
    return true;
}

} // namespace MR