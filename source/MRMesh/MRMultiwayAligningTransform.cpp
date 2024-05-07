#include "MRMultiwayAligningTransform.h"
#include <Eigen/Cholesky> //LLT
#include <cassert>

namespace MR
{

void MultiwayAligningTransform::reset( int numObjs )
{
    assert( numObjs >= 2 );
    int sz = ( numObjs - 1 ) * 6;
    a_.setZero( sz, sz );
    b_.setZero( sz );
    numObjs_ = numObjs;
}

void MultiwayAligningTransform::add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, const Vector3d& nB, double w )
{
    assert( objA >= 0 && objA < numObjs_ );
    assert( objB >= 0 && objB < numObjs_ );
    assert( objA != objB );

    double cA[6];
    cA[0] = nB.z * pA.y - nB.y * pA.z;
    cA[1] = nB.x * pA.z - nB.z * pA.x;
    cA[2] = nB.y * pA.x - nB.x * pA.y;
    cA[3] = nB.x;
    cA[4] = nB.y;
    cA[5] = nB.z;

    double cB[6];
    cB[0] = nB.y * pB.z - nB.z * pB.y;
    cB[1] = nB.z * pB.x - nB.x * pB.z;
    cB[2] = nB.x * pB.y - nB.y * pB.x;
    cB[3] = -nB.x;
    cB[4] = -nB.y;
    cB[5] = -nB.z;

    // update upper-right part of sumA_
    const double k_B = dot( pB - pA, nB );
    const int sA = objA * 6;
    const int sB = objB * 6;

    if ( objA + 1 < numObjs_ )
    {
        for ( size_t i = 0; i < 6; i++ )
        {
            for ( size_t j = i; j < 6; j++ )
                a_(sA+i, sA+j) += w * cA[i] * cA[j];
            b_(sA+i) += w * cA[i] * k_B;
        }
    }

    if ( objB + 1 < numObjs_ )
    {
        for ( size_t i = 0; i < 6; i++ )
        {
            for ( size_t j = i; j < 6; j++ )
                a_(sB+i, sB+j) += w * cB[i] * cB[j];
            b_(sB+i) += w * cB[i] * k_B;
        }
    }

    if ( objA + 1 < numObjs_ && objB + 1 < numObjs_ )
    {
        if ( objA < objB )
        {
            for ( size_t i = 0; i < 6; i++ )
                for ( size_t j = 0; j < 6; j++ )
                    a_(sA+i, sB+j) += w * cA[i] * cB[j];
        }
        else
        {
            for ( size_t i = 0; i < 6; i++ )
                for ( size_t j = 0; j < 6; j++ )
                    a_(sB+i, sA+j) += w * cB[i] * cA[j];
        }
    }
}

std::vector<RigidXf3d> MultiwayAligningTransform::solve()
{
    Eigen::LLT<Eigen::MatrixXd> chol( a_ );
    Eigen::VectorXd solution = chol.solve( b_ );

    std::vector<RigidXf3d> res;
    res.reserve( numObjs_ );
    for ( int i = 0; i + 1 < numObjs_; ++i )
    {
        RigidXf3d xf;
        int n = i * 6;
        xf.a = Vector3d{ solution.coeff( n+0 ), solution.coeff( n+1 ), solution.coeff( n+2 ) };
        xf.b = Vector3d{ solution.coeff( n+3 ), solution.coeff( n+4 ), solution.coeff( n+5 ) };
        res.push_back( std::move( xf ) );
    }
    res.emplace_back();
    assert( res.size() == numObjs_ );

    return res;
}

} //namespace MR
