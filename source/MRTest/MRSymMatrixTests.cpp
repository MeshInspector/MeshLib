#include <MRMesh/MRSymMatrix2.h>
#include <MRMesh/MRSymMatrix3.h>
#include <MRMesh/MRMatrix2.h>
#include <MRMesh/MRMatrix3.h>
#include <gtest/gtest.h>

namespace MR
{

// pseudoinverse inverts a well-conditioned positive definite matrix directly, other matrices via eigendecomposition
TEST( MRMesh, SymMatrix3Pseudoinverse )
{
    const double eps = std::numeric_limits<double>::epsilon();
    auto expectInverse = []( const SymMatrix3d & m, const SymMatrix3d & inv )
    {
        for ( const auto & v : { Vector3d( 1, 0, 0 ), Vector3d( 0, 1, 0 ), Vector3d( 0, 0, 1 ), Vector3d( 1, -2, 3 ) } )
            EXPECT_NEAR( ( inv * ( m * v ) - v ).length(), 0, 1e-9 * v.length() );
    };

    // positive definite: inverted directly
    SymMatrix3d pd;
    pd.xx = 4; pd.xy = 1; pd.xz = 0.5; pd.yy = 3; pd.yz = 0.2; pd.zz = 2;
    int rank = 0;
    Vector3d space( 1, 1, 1 );
    expectInverse( pd, pd.pseudoinverse( eps, &rank, &space ) );
    EXPECT_EQ( rank, 3 );
    EXPECT_EQ( space, Vector3d() );

    // indefinite with positive determinant: eigendecomposition, still the inverse
    SymMatrix3d ind;
    ind.xx = -1; ind.yy = -2; ind.zz = 3;
    expectInverse( ind, ind.pseudoinverse( eps, &rank ) );
    EXPECT_EQ( rank, 3 );

    // positive definite, but too ill-conditioned for the direct inverse: eigendecomposition, still the inverse
    SymMatrix3d ill = SymMatrix3d::diagonal( 1 );
    ill.zz = 1e-6;
    expectInverse( ill, ill.pseudoinverse( eps, &rank ) );
    EXPECT_EQ( rank, 3 );

    // the smallest eigenvalue below the tolerance is dropped even in a positive definite matrix
    SymMatrix3d small = SymMatrix3d::diagonal( 1 );
    small.zz = 1e-3;
    const auto smallInv = small.pseudoinverse( 1e-2, &rank, &space );
    EXPECT_EQ( rank, 2 );
    EXPECT_NEAR( std::abs( space.z ), 1, 1e-12 );
    EXPECT_NEAR( smallInv.zz, 0, 1e-12 );

    // positive semi-definite of rank 2
    SymMatrix3d psd;
    psd.xx = 2; psd.xy = 1; psd.yy = 2;
    const auto psdInv = psd.pseudoinverse( eps, &rank, &space );
    EXPECT_EQ( rank, 2 );
    EXPECT_NEAR( std::abs( space.z ), 1, 1e-12 );
    EXPECT_NEAR( ( psdInv * Vector3d( 0, 0, 1 ) ).length(), 0, 1e-12 );
}

TEST( MRMesh, SymMatrix2Pseudoinverse )
{
    const float eps = std::numeric_limits<float>::epsilon();
    int rank = 0;
    Vector2f space( 1, 1 );

    // positive definite: inverted directly
    SymMatrix2f pd;
    pd.xx = 3; pd.xy = 1; pd.yy = 2;
    const auto pdInv = pd.pseudoinverse( eps, &rank, &space );
    EXPECT_EQ( rank, 2 );
    EXPECT_EQ( space, Vector2f() );
    EXPECT_NEAR( ( pdInv * ( pd * Vector2f( 1, -2 ) ) - Vector2f( 1, -2 ) ).length(), 0, 1e-5f );

    // indefinite: eigendecomposition, still the inverse
    SymMatrix2f ind;
    ind.xx = 1; ind.xy = 2; ind.yy = 1;
    const auto indInv = ind.pseudoinverse( eps, &rank );
    EXPECT_EQ( rank, 2 );
    EXPECT_NEAR( ( indInv * ( ind * Vector2f( 1, -2 ) ) - Vector2f( 1, -2 ) ).length(), 0, 1e-5f );

    // rank 1
    SymMatrix2f psd;
    psd.xx = 1; psd.xy = 1; psd.yy = 1;
    const auto psdInv = psd.pseudoinverse( eps, &rank, &space );
    EXPECT_EQ( rank, 1 );
    EXPECT_NEAR( ( psdInv * Vector2f( 1, -1 ) ).length(), 0, 1e-6f );
}

} //namespace MR
