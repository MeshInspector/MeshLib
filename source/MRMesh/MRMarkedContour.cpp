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

MarkedContour3f resample( const MarkedContour3f & in, float minStep, Contour3f * normals )
{
    MR_TIMER;
    assert( !normals || normals->size() == in.contour.size() );
    MarkedContour3f res;
    if ( in.contour.empty() )
        return res;

    assert( firstLastMarked( in ) );
    res.marks.autoResizeSet( res.contour.size() );
    res.contour.push_back( in.contour.front() );
    Contour3f resNormals;
    if ( normals )
        resNormals.push_back( normals->front() );

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

        const int numMidPoints = int( distance / minStep );
        if ( numMidPoints > 0 )
        {
            const float step = distance / ( numMidPoints + 1 );
            float remDistance = step; //< till next sample point
            auto i2 = i;
            auto p = in.contour[i2];
            Vector3f n;
            if ( normals )
                n = ( *normals )[i2];
            while ( i2 < i1 )
            {
                auto segmLen = ( in.contour[i2 + 1] - p ).length();
                if ( segmLen <= remDistance )
                {
                    remDistance -= segmLen;
                    p = in.contour[++i2];
                    if ( normals )
                        n = ( *normals )[i2];
                    continue;
                }
                const float a = remDistance / segmLen;
                res.contour.push_back( p = ( 1 - a ) * p + a * in.contour[i2 + 1] );
                if ( normals )
                    resNormals.push_back( n = ( ( 1 - a ) * n + a * ( *normals )[i2 + 1] ).normalized() );
                remDistance = step;
            }
        }
        i = i1;
        res.marks.autoResizeSet( res.contour.size() );
        res.contour.push_back( in.contour[i] );
        if ( normals )
            resNormals.push_back( ( *normals )[i] );
    }

    assert( firstLastMarked( res ) );
    assert( in.marks.count() == res.marks.count() );
    if ( normals )
        *normals = std::move( resNormals );
    return res;
}

MarkedContour3f makeSpline( MarkedContour3f mc, float markStability )
{
    MR_TIMER;
    assert( markStability > 0 );
    if ( mc.contour.empty() )
        return mc;
    assert( firstLastMarked( mc ) );
    const bool closed = isClosed( mc.contour );

    const auto sz = mc.contour.size();
    const auto mz = mc.marks.count();

    std::vector<Eigen::Triplet<double>> mTriplets;
    const auto numVars = closed ? sz - 1 : sz;
    const auto numEqs = sz + mz - 2; // closed has one more smooth middle equation and one less marked point equation
    const auto nonZeros = closed ? 3 * ( sz - 1 ) + mz - 1 : 3 * ( sz - 2 ) + mz;
    mTriplets.reserve( nonZeros );

    Eigen::VectorXd r[3];
    for ( int i = 0; i < 3; ++i )
        r[i].resize( numEqs );
    auto setR = [&r]( int row, Vector3f p )
    {
        r[0][row] = p.x;
        r[1][row] = p.y;
        r[2][row] = p.z;
    };

    // Smoothness at middle points
    for ( int i = 0; i + 1 < sz; ++i )
        r[0][i] = r[1][i] = r[2][i] = 0;

    int nextRow = 0;
    for ( ; nextRow + 2 + closed < sz; ++nextRow )
    {
        mTriplets.emplace_back( nextRow, nextRow    , -0.5 );
        mTriplets.emplace_back( nextRow, nextRow + 1,  1.0 );
        mTriplets.emplace_back( nextRow, nextRow + 2, -0.5 );
    }
    if ( closed )
    {
        mTriplets.emplace_back( nextRow, int( sz - 3 ), -0.5 );
        mTriplets.emplace_back( nextRow, int( sz - 2 ),  1.0 );
        mTriplets.emplace_back( nextRow, 0,             -0.5 );
        ++nextRow;

        mTriplets.emplace_back( nextRow, int( sz - 2 ), -0.5 );
        mTriplets.emplace_back( nextRow, 0,              1.0 );
        mTriplets.emplace_back( nextRow, 1,             -0.5 );
        ++nextRow;
    }

    // Stabilization mc marked points
    for ( auto i : mc.marks )
    {
        if ( closed && i + 1 == sz )
            break;
        mTriplets.emplace_back( nextRow, int( i ), markStability );
        setR( nextRow, markStability * mc.contour[i] );
        ++nextRow;
    }
    assert( nextRow == numEqs );
    assert( mTriplets.size() == nonZeros );

    Eigen::SparseMatrix<double,Eigen::RowMajor> C;
    C.resize( numEqs, numVars );
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
    for ( int i = 0; i < numVars; ++i )
        mc.contour[i] = Vector3f( (float)x[0][i], (float)x[1][i], (float)x[2][i] );
    if ( closed )
        mc.contour.back() = mc.contour.front();
    return mc;
}

MarkedContour3f makeSpline( const Contour3f & controlPoints, const SplineSettings & settings )
{
    MR_TIMER;
    assert( settings.iterations >= 1 );
    MarkedContour3f res = markedContour( controlPoints );
    for( int i = 0; i < settings.iterations; ++i )
    {
        assert( controlPoints.size() == res.marks.count() );
        if ( i > 0 )
        {
            // restore exact control points positions
            int n = 0;
            for ( auto m : res.marks )
                res.contour[m] = controlPoints[n++];
        }
        res = makeSpline( resample( res, settings.samplingStep, settings.normals ), settings.controlStability );
    }
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
    EXPECT_EQ( spline.contour.size(), 4 );
}

TEST(MRMesh, MakeClosedSpline)
{
    Contour3f c{
        Vector3f{ 0, 0, 0 },
        Vector3f{ 1, 0, 0 },
        Vector3f{ 1, 1, 0 },
        Vector3f{ 0, 1, 0 },
        Vector3f{ 0, 0, 0 }
    };
    SplineSettings s{ .samplingStep = 0.4f };
    auto spline = makeSpline( c, s );
    EXPECT_EQ( spline.contour.size(), 13 );
    EXPECT_EQ( spline.contour.front(), spline.contour.back() );
    EXPECT_EQ( spline.marks.count(), 5 );
}

} //namespace MR
