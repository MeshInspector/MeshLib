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
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
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

MarkedContour3f makeSpline( MarkedContour3f mc, float markStability, const Contour3f * normals )
{
    if ( normals )
    {
        mc = makeSpline( std::move( mc ), *normals, markStability );
        return mc;
    }
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

MarkedContour3f makeSpline( MarkedContour3f mc, const Contour3f & normals, float markStability )
{
    MR_TIMER;
    assert( markStability > 0 );
    if ( mc.contour.empty() )
        return mc;
    assert( firstLastMarked( mc ) );
    const bool closed = isClosed( mc.contour );

    const auto sz = mc.contour.size();
    assert( sz == normals.size() );
    const auto mz = mc.marks.count();

    const auto numVars1 = int( closed ? sz - 1 : sz );
    const auto numVars = 3 * numVars1;
    const auto numEqs1 = sz + mz - 2;
    const auto numNormEqs = int( closed ? mz - 1 : mz );
    const auto numEqs = 3 * numEqs1 + numNormEqs;
    const auto nonZeros1 = closed ? 3 * ( sz - 1 ) + mz - 1 : 3 * ( sz - 2 ) + mz;
    const auto nonZeros = 3 * nonZeros1 + 6 * numNormEqs;

    std::vector<Eigen::Triplet<double>> mTriplets;
    mTriplets.reserve( nonZeros );
    auto addTriplet = [&mTriplets, numVars]( int r, int c, double v )
    {
        (void)numVars;
        assert( c >= 0 && c < numVars );
        mTriplets.emplace_back( r, c, v );
    };

    Eigen::VectorXd r;
    r.resize( numEqs );
    int nextRow = 0;

    // separable equations
    for ( int d = 0; d < 3; ++d )
    {
        const auto vd = d * (int)numVars1;

        // Smoothness at middle points
        for ( int i = 0; i + 2 + closed < sz; ++i )
        {
            addTriplet( nextRow, vd + i    , -0.5 );
            addTriplet( nextRow, vd + i + 1,  1.0 );
            addTriplet( nextRow, vd + i + 2, -0.5 );
            r[nextRow] = 0;
            ++nextRow;
        }
        if ( closed )
        {
            addTriplet( nextRow, vd + int( sz - 3 ), -0.5 );
            addTriplet( nextRow, vd + int( sz - 2 ),  1.0 );
            addTriplet( nextRow, vd + 0,             -0.5 );
            r[nextRow] = 0;
            ++nextRow;

            addTriplet( nextRow, vd + int( sz - 2 ), -0.5 );
            addTriplet( nextRow, vd + 0,              1.0 );
            addTriplet( nextRow, vd + 1,             -0.5 );
            r[nextRow] = 0;
            ++nextRow;
        }

        // Stabilization mc marked points
        for ( auto i : mc.marks )
        {
            if ( closed && i + 1 == sz )
                break;
            addTriplet( nextRow, vd + int( i ), markStability );
            r[nextRow] = markStability * mc.contour[i][d];
            ++nextRow;
        }

        assert( nextRow == ( d + 1 ) * numEqs1 );
        assert( mTriplets.size() == ( d + 1 ) * nonZeros1 );
    }

    // equations for normals at marked points
    auto addNorm = [&]( int in, int p0, int p1 )
    {
        auto n = normals[in];
        // dot( p1 - p0, n ) = 0
        addTriplet( nextRow, p0,                -n.x * markStability );
        addTriplet( nextRow, p0 + numVars1,     -n.y * markStability );
        addTriplet( nextRow, p0 + numVars1 * 2, -n.z * markStability );
        addTriplet( nextRow, p1,                 n.x * markStability );
        addTriplet( nextRow, p1 + numVars1,      n.y * markStability );
        addTriplet( nextRow, p1 + numVars1 * 2,  n.z * markStability );
        r[nextRow] = 0;
        ++nextRow;
    };

    // equations for normals at inner marked points
    for ( auto i : mc.marks )
    {
        if ( i == 0 )
        {
            addNorm( int(i), closed ? int(sz) - 2 : 0, 1 );
        }
        else if ( i + 1 == sz )
        {
            if ( closed )
                break;
            addNorm( int(i), int(i) - 1, int(i) );
        }
        else
        {
            if ( closed && i + 2 == sz )
                addNorm( int(i), int(i) - 1, 0 );
            else
                addNorm( int(i), int(i) - 1, int(i) + 1 );
        }
    }

    assert( nextRow == numEqs );
    assert( mTriplets.size() == nonZeros );

    Eigen::SparseMatrix<double,Eigen::RowMajor> C;
    C.resize( numEqs, numVars );
    C.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    // minimum squares solution:
    // C^T * C * x = C^T * rhs
    Eigen::SparseMatrix<double,Eigen::RowMajor> A = C.adjoint() * C;
    Eigen::VectorXd b = C.adjoint() * r;

    Eigen::SimplicialLDLT< Eigen::SparseMatrix<double,Eigen::ColMajor> > chol;
    chol.compute( A );
    Eigen::VectorXd x = chol.solve( b );

    // produce output
    for ( int i = 0; i < numVars1; ++i )
        mc.contour[i] = Vector3f( (float)x[i], (float)x[i + numVars1], (float)x[i + 2*numVars1] );
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
        res = makeSpline( resample( res, settings.samplingStep, settings.normals ), settings.controlStability,
            settings.normalsAffectShape ? settings.normals : nullptr );
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
