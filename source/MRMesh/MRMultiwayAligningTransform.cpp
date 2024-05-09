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

void MultiwayAligningTransform::add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, double w )
{
    assert( objA >= 0 && objA < numObjs_ );
    assert( objB >= 0 && objB < numObjs_ );
    assert( objA != objB );

    const Vector3d cA[6] =
    {
        {   0.0, -pA.z,  pA.y },
        {  pA.z,   0.0, -pA.x },
        { -pA.y,  pA.x,   0.0 },
        {   1.0,   0.0,   0.0 },
        {   0.0,   1.0,   0.0 },
        {   0.0,   0.0,   1.0 }
    };

    const Vector3d cB[6] =
    {
        {   0.0,  pB.z, -pB.y },
        { -pB.z,   0.0,  pB.x },
        {  pB.y, -pB.x,   0.0 },
        {  -1.0,   0.0,   0.0 },
        {   0.0,  -1.0,   0.0 },
        {   0.0,   0.0,  -1.0 }
    };

    // update upper-right part of sumA_
    const Vector3d k_B = pB - pA;
    const int sA = objA * 6;
    const int sB = objB * 6;

    if ( objA + 1 < numObjs_ )
    {
        for ( size_t i = 0; i < 6; i++ )
        {
            for ( size_t j = i; j < 6; j++ )
                a_(sA+i, sA+j) += w * dot( cA[i], cA[j] );
            b_(sA+i) += w * dot( cA[i], k_B );
        }
    }

    if ( objB + 1 < numObjs_ )
    {
        for ( size_t i = 0; i < 6; i++ )
        {
            for ( size_t j = i; j < 6; j++ )
                a_(sB+i, sB+j) += w * dot( cB[i], cB[j] );
            b_(sB+i) += w * dot( cB[i], k_B );
        }
    }

    if ( objA + 1 < numObjs_ && objB + 1 < numObjs_ )
    {
        if ( objA < objB )
        {
            for ( size_t i = 0; i < 6; i++ )
                for ( size_t j = 0; j < 6; j++ )
                    a_(sA+i, sB+j) += w * dot( cA[i], cB[j] );
        }
        else
        {
            for ( size_t i = 0; i < 6; i++ )
                for ( size_t j = 0; j < 6; j++ )
                    a_(sB+i, sA+j) += w * dot( cB[i], cA[j] );
        }
    }
}

void MultiwayAligningTransform::add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, const Vector3d& n, double w )
{
    assert( objA >= 0 && objA < numObjs_ );
    assert( objB >= 0 && objB < numObjs_ );
    assert( objA != objB );

    double cA[6];
    cA[0] = n.z * pA.y - n.y * pA.z;
    cA[1] = n.x * pA.z - n.z * pA.x;
    cA[2] = n.y * pA.x - n.x * pA.y;
    cA[3] = n.x;
    cA[4] = n.y;
    cA[5] = n.z;

    double cB[6];
    cB[0] = n.y * pB.z - n.z * pB.y;
    cB[1] = n.z * pB.x - n.x * pB.z;
    cB[2] = n.x * pB.y - n.y * pB.x;
    cB[3] = -n.x;
    cB[4] = -n.y;
    cB[5] = -n.z;

    // update upper-right part of sumA_
    const double k_B = dot( pB - pA, n );
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
    MultiwayAligningTransform mw1d, mw3d;

    // 2 objects, 0-1 links
    {
        mw1d.reset( 2 );
        mw3d.reset( 2 );
        for( int i = 0; i < 10; i++ )
        {
            mw1d.add( 0, pInit[i], 1, xf( pInit[i] ) + n2[i], n[i] );
            mw3d.add( 0, pInit[i], 1, xf( pInit[i] ) );
        }

        const auto xf1d = mw1d.solve()[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf1d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf1d.b ).length(), 0., eps );

        const auto xf3d = mw3d.solve()[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf3d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf3d.b ).length(), 0., eps );
    }

    // 2 objects, 1-0 links
    {
        mw1d.reset( 2 );
        mw3d.reset( 2 );
        for( int i = 0; i < 10; i++ )
        {
            mw1d.add( 1, xf( pInit[i] ) - n2[i], 0, pInit[i], n[i] );
            mw3d.add( 1, xf( pInit[i] ), 0, pInit[i] );
        }

        const auto xf1d = mw1d.solve()[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf1d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf1d.b ).length(), 0., eps );

        const auto xf3d = mw3d.solve()[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf3d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf3d.b ).length(), 0., eps );
    }

    const double alpha1 = -0.05, beta1 = -0.13, gamma1 = 0.07;
    const Vector3d eulerAngles1{ alpha1, beta1, gamma1 };
    const Matrix3d rotationMatrix1 = Matrix3d::approximateLinearRotationMatrixFromEuler( eulerAngles1 );
    const Vector3d b1( 1., -2., 3. );
    const AffineXf3d xf1( rotationMatrix1, b1 );

    // 3 objects: 0-2, 1-2 links
    {
        mw1d.reset( 3 );
        mw3d.reset( 3 );
        for( int i = 0; i < 10; i++ )
        {
            mw1d.add( 0, pInit[i], 2, xf( pInit[i] ) + n2[i], n[i] );
            mw3d.add( 0, pInit[i], 2, xf( pInit[i] ) );
            mw1d.add( 1, pInit[i], 2, xf1( pInit[i] ) - n2[i], n[i] );
            mw3d.add( 1, pInit[i], 2, xf1( pInit[i] ) );
        }

        const auto sol1d = mw1d.solve();
        const auto xf01d = sol1d[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf01d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf01d.b ).length(), 0., eps );
        auto xf11d = sol1d[1].linearXf();
        EXPECT_NEAR( ( xf1.A - xf11d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf11d.b ).length(), 0., eps );

        const auto sol3d = mw3d.solve();
        const auto xf03d = sol3d[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf03d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf03d.b ).length(), 0., eps );
        auto xf13d = sol3d[1].linearXf();
        EXPECT_NEAR( ( xf1.A - xf13d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf13d.b ).length(), 0., eps );
    }

    // 3 objects: 0-1, 1-2 links
    {
        mw1d.reset( 3 );
        mw3d.reset( 3 );
        for( int i = 0; i < 10; i++ )
        {
            // composition of xf01 and xf12 must be approximateLinearRotationMatrix, so xf01 is shift-only
            mw1d.add( 0, pInit[i], 1, pInit[i] + b1, n[i] );
            mw3d.add( 0, pInit[i], 1, pInit[i] + b1 );
            mw1d.add( 1, pInit[i], 2, xf( pInit[i] ) - n2[i], n[i] );
            mw3d.add( 1, pInit[i], 2, xf( pInit[i] ) );
        }
        const auto b0 = xf( b1 );

        const auto sol1d = mw1d.solve();
        auto xf01d = sol1d[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf01d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( b0 - xf01d.b ).length(), 0., 3 * eps );
        auto xf11d = sol1d[1].linearXf();
        EXPECT_NEAR( ( xf.A - xf11d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf11d.b ).length(), 0., 2 * eps );

        const auto sol3d = mw3d.solve();
        auto xf03d = sol3d[0].linearXf();
        EXPECT_NEAR( ( xf.A - xf03d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( b0 - xf03d.b ).length(), 0., 3 * eps );
        auto xf13d = sol3d[1].linearXf();
        EXPECT_NEAR( ( xf.A - xf13d.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf.b - xf13d.b ).length(), 0., 2 * eps );
    }
}

} //namespace MR
