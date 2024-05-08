#include "MRMultiwayAligningTransform.h"
#include "MRGTest.h"
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

void MultiwayAligningTransform::add( const MultiwayAligningTransform & r )
{
    assert( numObjs_ == r.numObjs_ );
    a_ += r.a_;
    b_ += r.b_;
}

std::vector<RigidXf3d> MultiwayAligningTransform::solve()
{
    // copy values in lower-left part
    const auto sz = (size_t)a_.rows();
    assert( sz == (size_t)a_.cols() );
    for (size_t i = 1; i < sz; i++)
        for (size_t j = 0; j < i; j++)
            a_(i, j) = a_(j, i);

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

TEST( MRMesh, MultiwayAligningTransform )
{
    std::vector<Vector3d> pInit, n, n2;
    pInit.resize( 10 );
    n.resize( 10 );
    n2.resize( 10 );

    pInit[0]  = {   1.0,   1.0, -5.0 }; n[0] = {  0.0,  0.0, -1.0 }; n2[0] = { 0.1, -0.1,  0.0 };
    pInit[1]  = {  14.0,   1.0,  1.0 }; n[1] = {  1.0,  0.1,  1.0 }; n2[1] = { 0.3,  0.0, -0.3 };
    pInit[2]  = {   1.0,  14.0,  2.0 }; n[2] = {  0.1,  1.0,  1.2 }; n2[2] = { 0.0, -0.6,  0.5 };
    pInit[3]  = { -11.0,   2.0,  3.0 }; n[3] = { -1.0,  0.1,  1.0 }; // other n2's are zero
    pInit[4]  = {   1.0, -11.0,  4.0 }; n[4] = {  0.1, -1.1,  1.1 };
    pInit[5]  = {   1.0,   2.0,  8.0 }; n[5] = {  0.1,  0.1,  1.0 };
    pInit[6]  = {   2.0,   1.0, -5.0 }; n[6] = {  0.1,  0.0, -1.0 };
    pInit[7]  = {  15.0,   1.5,  1.0 }; n[7] = {  1.1,  0.1,  1.0 };
    pInit[8]  = {   1.5,  15.0,  2.0 }; n[8] = {  0.1,  1.0,  1.2 };
    pInit[9]  = { -11.0,   2.5,  3.1 }; n[9] = { -1.1,  0.1,  1.1 };

    const double alpha = 0.15, beta = 0.23, gamma = -0.17;
    const Vector3d eulerAngles{ alpha, beta, gamma };
    const Matrix3d rotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( eulerAngles );
    const Vector3d b( 2., 3., -1. );
    const AffineXf3d xf( rotationMatrix, b );
    constexpr double eps = 3e-13;

    // 2 objects, 0-1 links
    {
        MultiwayAligningTransform mw;
        mw.reset( 2 );
        for( int i = 0; i < 10; i++ )
            mw.add( 0, pInit[i], 1, xf( pInit[i] ) + n2[i], n[i] );

        const auto ammendment = mw.solve()[0];
        auto xfT = ammendment.linearXf();
        EXPECT_NEAR( ( xf.A - xfT.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xfT.b ).length(), 0., eps );
    }

    // 2 objects, 1-0 links
    {
        MultiwayAligningTransform mw;
        mw.reset( 2 );
        for( int i = 0; i < 10; i++ )
            mw.add( 1, xf( pInit[i] ) - n2[i], 0, pInit[i], n[i] );

        const auto ammendment = mw.solve()[0];
        auto xfT = ammendment.linearXf();
        EXPECT_NEAR( ( xf.A - xfT.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xfT.b ).length(), 0., eps );
    }

    const double alpha1 = -0.05, beta1 = -0.13, gamma1 = 0.07;
    const Vector3d eulerAngles1{ alpha1, beta1, gamma1 };
    const Matrix3d rotationMatrix1 = Matrix3d::approximateLinearRotationMatrixFromEuler( eulerAngles1 );
    const Vector3d b1( 1., -2., 3. );
    const AffineXf3d xf1( rotationMatrix1, b1 );

    // 3 objects: 0-2, 1-2 links
    {
        MultiwayAligningTransform mw;
        mw.reset( 3 );
        for( int i = 0; i < 10; i++ )
        {
            mw.add( 0, pInit[i], 2, xf( pInit[i] ) + n2[i], n[i] );
            mw.add( 1, pInit[i], 2, xf1( pInit[i] ) - n2[i], n[i] );
        }
        const auto sol = mw.solve();

        auto xfT0 = sol[0].linearXf();
        EXPECT_NEAR( ( xf.A - xfT0.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xfT0.b ).length(), 0., eps );

        auto xfT1 = sol[1].linearXf();
        EXPECT_NEAR( ( xf1.A - xfT1.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xfT1.b ).length(), 0., eps );
    }

    // 3 objects: 0-1, 1-2 links
    {
        MultiwayAligningTransform mw;
        mw.reset( 3 );
        for( int i = 0; i < 10; i++ )
        {
            mw.add( 0, pInit[i], 1, pInit[i], n[i] );
            mw.add( 1, pInit[i], 2, xf( pInit[i] ) - n2[i], n[i] );
        }
        const auto sol = mw.solve();

        auto xfT0 = sol[0].linearXf();
        EXPECT_NEAR( ( xf.A - xfT0.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xfT0.b ).length(), 0., eps );

        auto xfT1 = sol[1].linearXf();
        EXPECT_NEAR( ( xf.A - xfT1.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xfT1.b ).length(), 0., eps );
    }
}

} //namespace MR
