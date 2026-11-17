#pragma once

#include "MRMatrix3.h"

namespace MR
{

/// Decomposes matrix into rotation and scaling matrices;
/// if m is a mirror (det < 0) then by default all scales are non-negative and the rotation is improper (det = -1),
/// and with \p signedScale the rotation is proper (det = +1) and one scale is negative,
/// the one that leaves the rotation closest to identity (so an axis mirror gives the identity rotation)
template <typename T>
void decomposeMatrix3( const Matrix3<T>& m, Matrix3<T>& rotation, Matrix3<T>& scaling, bool signedScale = false )
{
    const auto [q, r] = m.qr();

    scaling = {};
    Matrix3<T> sign;
    for ( int i = 0; i < 3; ++i )
    {
        scaling[i][i] = std::abs( r[i][i] );
        if ( r[i][i] < 0 )
            sign[i][i] = -1;
    }
    rotation = q * sign;
    if ( signedScale && rotation.det() < 0 )
    {
        // negating the column with the smallest diagonal element gives the largest trace
        int j = 0;
        for ( int i = 1; i < 3; ++i )
            if ( rotation[i][i] < rotation[j][j] )
                j = i;
        for ( int i = 0; i < 3; ++i )
            rotation[i][j] = -rotation[i][j];
        scaling[j][j] = -scaling[j][j];
    }
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
