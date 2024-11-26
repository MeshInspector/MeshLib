#include "MRMarkedContour.h"
#include "MRTimer.h"
#include "MRGTest.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

namespace
{

[[maybe_unused]] bool firstLastMarked( const MarkedContour3f & in )
{
    if ( !in.marks.test( 0 ) )
        return false;

    if ( in.marks.find_last() + 1 != in.contour.size() )
        return false;

    return true;
}

} // anonymous namespace

MarkedContour3f resample( const MarkedContour3f & in, float maxStep )
{
    MR_TIMER
    MarkedContour3f res;
    if ( in.contour.empty() )
        return res;

    assert( firstLastMarked( in ) );
    res.marks.autoResizeSet( res.contour.size() );
    res.contour.push_back( in.contour.front() );

    for ( size_t i = 0; i + 1 < in.contour.size(); )
    {
        auto i1 = i; //< will be next mark
        float distance = 0; //< distance to next mark
        for ( ++i1; i1 < in.contour.size(); ++i1 )
        {
            distance += ( in.contour[i1 - 1] - in.contour[i1] ).length();
            if ( in.marks.test( i1 ) )
                break;
        }
        assert( in.marks.test( i1 ) );

        const int numMidPoints = int( distance / maxStep );
        if ( numMidPoints > 0 )
        {
            const float step = distance / ( numMidPoints + 1 );
            float remDistance = step; //< till next sample point
            auto i2 = i;
            auto p = in.contour[i2];
            while ( i2 < i1 )
            {
                auto segmLen = ( in.contour[i2 + 1] - p ).length();
                if ( segmLen <= remDistance )
                {
                    remDistance -= segmLen;
                    p = in.contour[++i2];
                    continue;
                }
                const float a = remDistance / segmLen;
                p = ( 1 - a ) * p + a * in.contour[i2 + 1];
                res.contour.push_back( p );
                remDistance = step;
            }
        }
        i = i1;
        res.marks.autoResizeSet( res.contour.size() );
        res.contour.push_back( in.contour[i] );
    }

    assert( firstLastMarked( res ) );
    assert( in.marks.count() == res.marks.count() );
    return res;
}

MarkedContour3f makeSpline( const MarkedContour3f & in, float markStability )
{
    MR_TIMER
    assert( markStability > 0 );
    MarkedContour3f res;
    if ( in.contour.empty() )
        return res;
    assert( firstLastMarked( in ) );

    const auto sz = in.contour.size();
    const auto mz = in.marks.count();

    std::vector<Eigen::Triplet<double>> mTriplets;
    mTriplets.reserve( 3 * ( sz - 2 ) + mz );

    Eigen::VectorXd r[3];
    for ( int i = 0; i < 3; ++i )
        r[i].resize( sz - 2 + mz );
    auto setR = [&r]( int row, Vector3f p )
    {
        r[0][row] = p.x;
        r[1][row] = p.y;
        r[2][row] = p.z;
    };

    // Smoothness at middle points
    for ( int i = 0; i + 2 < sz; ++i )
        r[0][i] = r[1][i] = r[2][i] = 0;

    for ( int i = 0; i + 2 < sz; ++i )
    {
        mTriplets.emplace_back( i, i    , -0.5 );
        mTriplets.emplace_back( i, i + 1,  1.0 );
        mTriplets.emplace_back( i, i + 2, -0.5 );
    }

    // Stabilization in marked points
    int nextRow = int( sz - 2 );
    for ( auto i : in.marks )
    {
        mTriplets.emplace_back( nextRow, int( i ), markStability );
        setR( nextRow, markStability * in.contour[i] );
        ++nextRow;
    }

    Eigen::SparseMatrix<double,Eigen::RowMajor> C;
    C.resize( sz - 2 + mz, sz );
    assert( mTriplets.size() == 3 * ( sz - 2 ) + mz );
    C.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    // minimum squares solution:
    // C^T * C * x = C^T * rhs
    Eigen::SparseMatrix<double,Eigen::RowMajor> A = C.adjoint() * C;
    Eigen::VectorXd b[3];
    for ( int i = 0; i < 3; ++i )
        b[i] = C.adjoint() * r[i];

    Eigen::SimplicialLDLT< Eigen::SparseMatrix<double,Eigen::ColMajor> > chol;
    chol.compute( A );
    Eigen::VectorXd x[3];
    x[0] = chol.solve( b[0] );
    x[1] = chol.solve( b[1] );
    x[2] = chol.solve( b[2] );

    // produce output
    res.marks = in.marks;
    res.contour.reserve( sz );
    for ( int i = 0; i < sz; ++i )
        res.contour.push_back( Vector3f( (float)x[0][i], (float)x[1][i], (float)x[2][i] ) );
    return res;
}

TEST(MRMesh, MarkedContour)
{
    auto mc = markedContour( Contour3f{ Vector3f{ 0, 0, 0 }, Vector3f{ 1, 0, 0 } } );
    auto rc = resample( mc, 2 );
    EXPECT_EQ( mc.contour, rc.contour );
    EXPECT_EQ( mc.marks, rc.marks );

    rc = resample( mc, 0.4f );
    EXPECT_EQ( rc.contour.size(), 4 );

    auto spline = makeSpline( rc );
}

} //namespace MR
