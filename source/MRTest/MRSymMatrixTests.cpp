#include <MRMesh/MRSymMatrix2.h>
#include <MRMesh/MRSymMatrix3.h>
#include <MRMesh/MRSymMatrix4.h>
#include <MRMesh/MRMatrix2.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMatrix4.h>
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

// pseudoinverse of ill-conditioned and singular matrices relies on the precision of eigensJacobi for small eigenvalues
TEST( MRMesh, SymMatrix3PseudoinversePrecision )
{
    const double eps = std::numeric_limits<double>::epsilon();

    // least squares circle |p|^2 = 2*dot( p, c ) + r^2 - |c|^2 through 1 degree of arc, in the coordinates relative to the middle of the arc
    const Vector2d c( -299.5, 20 );
    const double r = c.length();
    const double a0 = std::atan2( -c.y, -c.x );
    SymMatrix3d m;
    Vector3d rhs;
    for ( int i = -10; i <= 10; ++i )
    {
        const double a = a0 + i * ( 0.5 * PI / 180 ) / 10;
        const Vector2d p = c + r * Vector2d( std::cos( a ), std::sin( a ) );
        const Vector3d row( 2 * p.x, 2 * p.y, 1 );
        m += outerSquare( row );
        rhs += row * p.lengthSq();
    }
    int rank = 0;
    const auto sol = m.pseudoinverse( eps, &rank ) * rhs;
    EXPECT_EQ( rank, 3 );
    EXPECT_NEAR( ( Vector2d( sol.x, sol.y ) - c ).length(), 0, 3e-9 );
    EXPECT_NEAR( std::sqrt( sol.z + sqr( sol.x ) + sqr( sol.y ) ), r, 3e-9 );

    // exactly singular matrices in general position
    const Vector3d u( 1, 1, 1 ), v( 1, 2, 4 ), w( 10, 1, 0.1 );
    const auto rank2 = outerSquare( u ) + outerSquare( v );
    Vector3d space;
    const auto inv2 = rank2.pseudoinverse( eps, &rank, &space );
    EXPECT_EQ( rank, 2 );
    EXPECT_NEAR( cross( space, cross( u, v ).normalized() ).length(), 0, 1e-12 );
    for ( const auto & x : { u, v } )
        EXPECT_NEAR( ( inv2 * ( rank2 * x ) - x ).length(), 0, 1e-12 );

    const auto rank1 = outerSquare( w );
    const auto inv1 = rank1.pseudoinverse( eps, &rank, &space );
    EXPECT_EQ( rank, 1 );
    EXPECT_NEAR( cross( space, w.normalized() ).length(), 0, 1e-12 );
    EXPECT_NEAR( ( inv1 * ( rank1 * w ) - w ).length(), 0, 1e-12 );

    // the determinant and the minors of a rank-1 matrix are rounding errors, which must not let it be inverted directly;
    // the pseudoinverse of outerSquare( n ) for unit n is the same matrix
    for ( int i = 0; i < 1000; ++i )
    {
        const auto n = Vector3d( std::sin( 1.1 * i ), std::cos( 1.7 * i ), std::sin( 2.3 * i + 1 ) ).normalized();
        const auto m1 = outerSquare( n );
        for ( double tol : { eps, 0.01 } )
        {
            EXPECT_NEAR( ( m1.pseudoinverse( tol, &rank ) * n - n ).length(), 0, 1e-12 );
            EXPECT_EQ( rank, 1 );
        }
    }
}

// eigensJacobi returns the same as eigens, with small eigenvalues found much more precisely
TEST( MRMesh, SymMatrix3EigensJacobi )
{
    SymMatrix3d m;
    m.xx = 4; m.xy = 1; m.xz = 0.5; m.yy = 3; m.yz = 0.2; m.zz = 2;
    Matrix3d vecs, vecsJ;
    const auto vals = m.eigens( &vecs );
    const auto valsJ = m.eigensJacobi( &vecsJ );
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_NEAR( vals[i], valsJ[i], 1e-14 );
        EXPECT_NEAR( std::abs( dot( vecs[i], vecsJ[i] ) ), 1, 1e-14 );
        EXPECT_NEAR( ( m * vecsJ[i] - valsJ[i] * vecsJ[i] ).length(), 0, 1e-14 );
    }

    // diag( 1, 1e-6, 1e-12 ) scaled as D*C*D with a well-conditioned C: all eigenvalues keep their relative precision
    SymMatrix3d g;
    g.xx = 1; g.xy = 0.5e-3; g.xz = 0.5e-6; g.yy = 1e-6; g.yz = 0.5e-9; g.zz = 1e-12;
    const auto e = g.eigensJacobi();
    EXPECT_NEAR( e[0], 2e-12 / 3, 1e-18 );
    EXPECT_NEAR( e[1], 0.75e-6, 1e-12 );
    EXPECT_NEAR( e[0] * e[1] * e[2], g.det(), 1e-12 * g.det() );
}

// inverse of integer matrices divides each element by the determinant
TEST( MRMesh, SymMatrixIntegerInverse )
{
    SymMatrix2i m2;
    m2.xx = 4; m2.xy = 2; m2.yy = 2; // det = 4
    const auto i2 = m2.inverse();
    EXPECT_EQ( i2.xx, 0 ); // 2 / 4
    EXPECT_EQ( i2.xy, 0 ); // -2 / 4
    EXPECT_EQ( i2.yy, 1 ); // 4 / 4

    SymMatrix3i m3 = SymMatrix3i::diagonal( 2 ); // det = 8, cofactors 4
    m3.zz = 4; // det = 16, cofactors xx = yy = 8, zz = 4
    const auto i3 = m3.inverse();
    EXPECT_EQ( i3.xx, 0 ); // 8 / 16
    EXPECT_EQ( i3.zz, 0 ); // 4 / 16
    EXPECT_EQ( m3.inverse( 4 ).xx, 2 ); // 8 / 4 with a given determinant
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

// all 10 elements take part in the arithmetic operators
TEST( MRMesh, SymMatrix4Operators )
{
    auto m = SymMatrix4d::identity();
    m += SymMatrix4d::diagonal( 2 );
    EXPECT_EQ( m, SymMatrix4d::diagonal( 3 ) );
    m -= SymMatrix4d::identity();
    EXPECT_EQ( m, SymMatrix4d::diagonal( 2 ) );
    m *= 4.0;
    EXPECT_EQ( m, SymMatrix4d::diagonal( 8 ) );
    m /= 2.0;
    EXPECT_EQ( m, SymMatrix4d::diagonal( 4 ) );
    auto mi = SymMatrix4i::diagonal( 6 );
    mi /= 3;
    EXPECT_EQ( mi, SymMatrix4i::diagonal( 2 ) );
}

TEST( MRMesh, SymMatrix4Eigens )
{
    auto check = []( const SymMatrix4d & m, double scale )
    {
        Matrix4d vecs;
        const auto vals = m.eigens( &vecs );
        EXPECT_EQ( vals, m.eigens() );
        for ( int i = 0; i < 4; ++i )
        {
            if ( i > 0 )
            {
                EXPECT_LE( vals[i - 1], vals[i] );
            }
            EXPECT_NEAR( ( m * vecs[i] - vals[i] * vecs[i] ).length(), 0, 1e-14 * scale );
            for ( int j = 0; j < 4; ++j )
                EXPECT_NEAR( dot( vecs[i], vecs[j] ), i == j ? 1 : 0, 1e-14 );
        }
        return vals;
    };

    auto vals = check( SymMatrix4d::diagonal( 5 ), 5 );
    EXPECT_EQ( vals, Vector4d( 5, 5, 5, 5 ) );

    SymMatrix4d m;
    m.xx = 4; m.xy = 1; m.xz = -2; m.xw = 2;
              m.yy = 2; m.yz = 0;  m.yw = 1;
                        m.zz = 3;  m.zw = -2;
                                   m.ww = -1;
    vals = check( m, 10 );
    EXPECT_NEAR( vals[0] + vals[1] + vals[2] + vals[3], m.trace(), 1e-13 );
    EXPECT_NEAR( sqr( vals[0] ) + sqr( vals[1] ) + sqr( vals[2] ) + sqr( vals[3] ), m.normSq(), 1e-12 );

    // two tiny and two close eigenvalues: 1e-12, 2e-12, 1, 1 + 1e-9 in a rotated basis
    const Vector4d dirs[4] = { Vector4d( 1, 1, 1, 1 ) / 2.0, Vector4d( 1, -1, 1, -1 ) / 2.0, Vector4d( 1, 1, -1, -1 ) / 2.0, Vector4d( 1, -1, -1, 1 ) / 2.0 };
    const double lambdas[4] = { 1e-12, 2e-12, 1, 1 + 1e-9 };
    SymMatrix4d t;
    for ( int i = 0; i < 4; ++i )
        t += outerSquare( lambdas[i], dirs[i] );
    vals = check( t, 1 );
    for ( int i = 0; i < 4; ++i )
        EXPECT_NEAR( vals[i], lambdas[i], 1e-14 );
    Matrix4d vecs;
    t.eigens( &vecs );
    EXPECT_NEAR( std::abs( dot( vecs[0], dirs[0] ) ), 1, 1e-3 );
    EXPECT_NEAR( std::abs( dot( vecs[1], dirs[1] ) ), 1, 1e-3 );
}

TEST( MRMesh, SymMatrix4Pseudoinverse )
{
    const double eps = std::numeric_limits<double>::epsilon();
    SymMatrix4d m;
    m.xx = 4; m.xy = 1; m.xz = -2; m.xw = 2;
              m.yy = 2; m.yz = 0;  m.yw = 1;
                        m.zz = 3;  m.zw = -2;
                                   m.ww = -1;
    int rank = 0;
    const auto inv = m.pseudoinverse( eps, &rank );
    EXPECT_EQ( rank, 4 );
    for ( const auto & v : { Vector4d( 1, 0, 0, 0 ), Vector4d( 0, 1, 0, 0 ), Vector4d( 0, 0, 1, 0 ), Vector4d( 0, 0, 0, 1 ), Vector4d( 1, -2, 3, -4 ) } )
        EXPECT_NEAR( ( inv * ( m * v ) - v ).length(), 0, 1e-12 * v.length() );

    // positive semi-definite of rank 2: the kernel is mapped to zero, the rest is inverted
    const Vector4d a( 1, 2, 0, -1 ), b( 0, 1, 1, 1 );
    const auto psd = outerSquare( a ) + outerSquare( b );
    const auto psdInv = psd.pseudoinverse( 1e-12, &rank );
    EXPECT_EQ( rank, 2 );
    EXPECT_NEAR( ( psdInv * Vector4d( 2, -1, 1, 0 ) ).length(), 0, 1e-12 ); // orthogonal to a and b
    for ( const auto & v : { a, b, a - 3.0 * b } )
        EXPECT_NEAR( ( psdInv * ( psd * v ) - v ).length(), 0, 1e-12 * v.length() );
}

TEST( MRMesh, SymMatrix4SolveLDLT )
{
    // positive definite: A^T*A + I
    SymMatrix4d m = SymMatrix4d::identity();
    for ( const auto & r : { Vector4d( 1, 2, 0, -1 ), Vector4d( 0, 1, 1, 1 ), Vector4d( 3, -1, 2, 0 ), Vector4d( -2, 0, 1, 4 ) } )
        m += outerSquare( r );
    for ( const auto & x : { Vector4d( 1, 0, 0, 0 ), Vector4d( 0, 0, 0, 1 ), Vector4d( 1, -2, 3, -4 ), Vector4d( 1e3, 1e-3, -5, 0.5 ) } )
        EXPECT_NEAR( ( m.solveLDLT( m * x ) - x ).length(), 0, 1e-12 * x.length() );

    // same as pseudoinverse for a positive definite matrix
    const Vector4d b( 0.3, -1, 2, 7 );
    EXPECT_NEAR( ( m.solveLDLT( b ) - m.pseudoinverse() * b ).length(), 0, 1e-12 * b.length() );

    // positive semi-definite with exactly zero pivots: finite solution of a consistent system
    const auto psd = outerSquare( Vector4d( 1, 1, 0, 0 ) ) + outerSquare( Vector4d( 0, 0, 1, 0 ) ) + outerSquare( Vector4d( 0, 0, 0, 1 ) );
    const Vector4d bp( 1, 1, 2, 3 );
    const auto xp = psd.solveLDLT( bp );
    EXPECT_NEAR( ( psd * xp - bp ).length(), 0, 1e-12 );
    EXPECT_EQ( SymMatrix4d{}.solveLDLT( Vector4d( 1, 2, 3, 4 ) ), Vector4d() );
}

} //namespace MR
