#include "MRFreeFormDeformer.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRPch/MRTBB.h"
#include <span>

// unknown pragmas
#pragma warning(disable:4068)

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <Eigen/Eigenvalues>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace
{
using namespace MR;
Vector3f interpolateNPoints( const std::span<const Vector3f>& points, float coef )
{
    const float invCoef = 1 - coef;
    if ( points.size() == 2 )
        return points[0] * invCoef + points[1] * coef;

    const size_t N = points.size();
    const size_t tempPointCount = N * ( N - 1 ) / 2 - 1;

    std::array<Vector3f, 14> tempPoints;   

    for ( size_t i = 0; i < N - 1; ++i )
    {
        tempPoints[i] = points[i] * invCoef + points[i + 1] * coef;
    }

    size_t offset = 0;

    for ( size_t j = N - 1; j > 2; --j )
    {
        for ( size_t i = 0; i < j - 1; ++i )
            tempPoints[offset + j + i] = tempPoints[offset + i] * invCoef + tempPoints[offset + i + 1] * coef;

        offset += j;
    }

    return tempPoints[offset] * invCoef + tempPoints[offset + 1] * coef;
 /*   auto pointsCopy = points;
    std::vector<Vector3f> nextStepPoints( points.size() - 1 );
    while ( !nextStepPoints.empty() )
    {
        for ( int i = 0; i < nextStepPoints.size(); ++i )
            nextStepPoints[i] = pointsCopy[i] * ( 1 - coef ) + pointsCopy[i + 1] * coef;

        pointsCopy = nextStepPoints;
        nextStepPoints.resize( pointsCopy.size() - 1 );
    }
    assert( pointsCopy.size() == 1 );*/
}

std::vector<Vector3f> makeOriginGrid( const Box3f& box, const Vector3i& resolution )
{
    auto resXY = resolution.x * resolution.y;
    std::vector<Vector3f> res( resXY *resolution.z );
    for ( int z = 0; z < resolution.z; ++z )
    for ( int y = 0; y < resolution.y; ++y )
    for ( int x = 0; x < resolution.x; ++x )
    {
        auto index = x + y * resolution.x + z * resXY;
        Vector3f coef( x / float( resolution.x - 1 ), y / float( resolution.y - 1 ), z / float( resolution.z - 1 ) );

        res[index] = Vector3f(
            box.min.x * ( 1.0f - coef.x ) + box.max.x * coef.x,
            box.min.y * ( 1.0f - coef.y ) + box.max.y * coef.y,
            box.min.z * ( 1.0f - coef.z ) + box.max.z * coef.z );
    }
    return res;
}

// simple factorial func
int factorial( int n )
{
    if ( n == 0 || n == 1 )
        return 1;
    int res = n;
    for ( int i = n - 1; i > 1; --i )
        res *= i;
    return res;
}

//             n!
// C(n,k) = --------
//          k!(n-k)!
int combination( int n, int k )
{
    return factorial( n ) / ( factorial( k ) * factorial( n - k ) );
}

// 0:       1
// 1:      1 1
// 2:     1 2 1
// 3:    1 3 3 1
// 4:   1 4 6 4 1
// .................
// line n pos k value is C(n,k)
std::vector<int> getPascalTriangleLine( int line )
{
    assert( line >= 0 );
    std::vector<int> res( line+1 );
    res[0] = 1;
    for ( int i = 1; i <= line / 2; ++i )
        res[i] = combination( line, i );
    for ( int i = line; i > line / 2; --i )
        res[i] = res[line - i];
    return res;
}

// simple pow function, not to use slow std::pow
float cyclePow( float a, int b )
{
    if ( b == 0 )
        return 1.0f;
    float res = a;
    for ( int i = 1; i < b; ++i )
        res *= a;
    return res;
}

// Optimized by passing precalculated parameters
std::vector<float> freeformWeights( const std::vector<int>& pascalLineX,
                                    const std::vector<int>& pascalLineY,
                                    const std::vector<int>& pascalLineZ,
                                    const Vector3f& boxMin,
                                    const Vector3f& backDiagonal,// 1 - box.diagonal
                                    const int resXY,
                                    const Vector3f& point )
{
    Vector3i resolution( int( pascalLineX.size() ), int( pascalLineY.size() ), int( pascalLineZ.size() ) );
    std::vector<float> result( resXY * resolution.z );

    std::vector<float> xCoefs( resolution.x );
    std::vector<float> yCoefs( resolution.y );
    std::vector<float> zCoefs( resolution.z );

    auto minToPoint = point - boxMin;
    auto pointRatio = Vector3f(
        minToPoint.x * backDiagonal.x,
        minToPoint.y * backDiagonal.y,
        minToPoint.z * backDiagonal.z );
    auto backPointRatio = Vector3f::diagonal( 1.0f ) - pointRatio;

    // Bezier curves reverse, to find each point weight
    for ( int xi = 0; xi < resolution.x; ++xi )
        xCoefs[xi] = pascalLineX[xi] * cyclePow( pointRatio.x, xi ) * cyclePow( backPointRatio.x, resolution.x - xi - 1 );
    for ( int yi = 0; yi < resolution.y; ++yi )
        yCoefs[yi] = pascalLineY[yi] * cyclePow( pointRatio.y, yi ) * cyclePow( backPointRatio.y, resolution.y - yi - 1 );
    for ( int zi = 0; zi < resolution.z; ++zi )
        zCoefs[zi] = pascalLineZ[zi] * cyclePow( pointRatio.z, zi ) * cyclePow( backPointRatio.z, resolution.z - zi - 1 );

    for ( int x = 0; x < resolution.x; ++x )
    for ( int y = 0; y < resolution.y; ++y )
    for ( int z = 0; z < resolution.z; ++z )
    {
        int index = x + y * resolution.x + z * resXY;
        result[index] = xCoefs[x] * yCoefs[y] * zCoefs[z];
    }
    return result;
}

}

namespace MR
{

FreeFormDeformer::FreeFormDeformer( Mesh& mesh ) :
    mesh_{mesh}
{
}

void FreeFormDeformer::init( const Vector3i& resolution /*= Vector3i::diagonal( 2 ) */, const Box3f& initialBox /*= Box3f() */ )
{
    assert( resolution.x > 1 && resolution.y > 1 && resolution.z > 1 );
    initialBox_ = initialBox.valid() ? initialBox : mesh_.computeBoundingBox();
    const auto& meshPoints = mesh_.points;
    meshPointsNormedPoses_.resize( meshPoints.size() );
    auto diagonalVec = initialBox_.max - initialBox_.min;
    diagonalVec.x = 1.0f / diagonalVec.x;
    diagonalVec.y = 1.0f / diagonalVec.y;
    diagonalVec.z = 1.0f / diagonalVec.z;
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)meshPointsNormedPoses_.size() ),
        [&]( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            VertId vid = VertId( i );
            const auto& point = meshPoints[vid];
            auto minToPoint = point - initialBox_.min;
            meshPointsNormedPoses_[i] = Vector3f(
                minToPoint.x * diagonalVec.x,
                minToPoint.y * diagonalVec.y,
                minToPoint.z * diagonalVec.z );
        }
    } );

    resolution_ = resolution;
    refPointsGrid_ = makeOriginGrid( initialBox_, resolution );
}

void FreeFormDeformer::setRefGridPointPosition( const Vector3i& coordOfPointInGrid, const Vector3f& newPos )
{
    refPointsGrid_[getIndex( coordOfPointInGrid )] = newPos;
}

const Vector3f& FreeFormDeformer::getRefGridPointPosition( const Vector3i& coordOfPointInGrid ) const
{
    return refPointsGrid_[getIndex( coordOfPointInGrid )];
}

void FreeFormDeformer::apply() const
{
    auto& meshPoints = mesh_.points;
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int) meshPoints.size() ),
        [&]( const tbb::blocked_range<int>& range )
    {
        std::vector<Vector3f> xPlane( resolution_.y * resolution_.z );
        std::vector<Vector3f> yLine( resolution_.z );
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            meshPoints[VertId( i )] = applyToNormedPoint_( meshPointsNormedPoses_[i], xPlane, yLine );
        }
    } );
}

Vector3f FreeFormDeformer::applySinglePoint( const Vector3f& point ) const
{
    auto diagonalVec = initialBox_.max - initialBox_.min;
    diagonalVec.x = 1.0f / diagonalVec.x;
    diagonalVec.y = 1.0f / diagonalVec.y;
    diagonalVec.z = 1.0f / diagonalVec.z;

    Vector3f normedPoint = point - initialBox_.min;
    normedPoint = Vector3f( normedPoint.x * diagonalVec.x, normedPoint.y * diagonalVec.y, normedPoint.z * diagonalVec.z );

    std::vector<Vector3f> xPlane( resolution_.y * resolution_.z );
    std::vector<Vector3f> yLine( resolution_.z );

    return applyToNormedPoint_( normedPoint, xPlane, yLine );
}

int FreeFormDeformer::getIndex( const Vector3i& coordOfPointInGrid ) const
{
    return coordOfPointInGrid.x + coordOfPointInGrid.y * resolution_.x + coordOfPointInGrid.z * resolution_.x * resolution_.y;
}

Vector3i FreeFormDeformer::getCoord( int index ) const
{
    Vector3i res;
    int resXY = resolution_.x * resolution_.y;
    res.z = index / resXY;
    int subZ = index % resXY;
    res.y = subZ / resolution_.x;
    res.x = subZ % resolution_.x;
    return res;
}

Vector3f FreeFormDeformer::applyToNormedPoint_( const Vector3f& normedPoint, std::vector<Vector3f>& xPlaneCache, std::vector<Vector3f>& yLineCache ) const
{
    for ( int z = 0; z < resolution_.z; ++z )
        for ( int y = 0; y < resolution_.y; ++y )
        {
            auto index = y + z * resolution_.y;
            auto indexWithX = index * resolution_.x;
            xPlaneCache[index] = interpolateNPoints( { refPointsGrid_.data() + indexWithX, size_t(resolution_.x) }, normedPoint.x );
        }

    for ( int z = 0; z < resolution_.z; ++z )
    {
        yLineCache[z] = interpolateNPoints( { xPlaneCache.begin() + z * resolution_.y, size_t( resolution_.y ) }, normedPoint.y );
    }
    return interpolateNPoints( yLineCache, normedPoint.z );
}

std::vector<Vector3f> findBestFreeformDeformation( const Box3f& box, const std::vector<Vector3f>& source, const std::vector<Vector3f>& target, 
                                                   const Vector3i& resolution /*= Vector3i::diagonal( 2 ) */ )
{
    // This parameters are needed to optimize freeform weights calculations
    auto resXY = resolution.x * resolution.y;
    auto pascalLineX = getPascalTriangleLine( resolution.x - 1 );
    auto pascalLineY = getPascalTriangleLine( resolution.y - 1 );
    auto pascalLineZ = getPascalTriangleLine( resolution.z - 1 );

    auto refPointsCout = resXY * resolution.z;

    auto backDiagonal = box.max - box.min;
    backDiagonal.x = 1.0f / backDiagonal.x;
    backDiagonal.y = 1.0f / backDiagonal.y;
    backDiagonal.z = 1.0f / backDiagonal.z;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A( refPointsCout, refPointsCout );
    Eigen::Matrix<double, Eigen::Dynamic, 3> B( refPointsCout, 3 );
    A.setZero();
    B.setZero();

    auto vecToEigen = []( const Vector3f& vec )
    {
        return Eigen::Vector3d( vec[0], vec[1], vec[2] );
    };

    auto eigenToVec = []( const Eigen::Vector3d& vec )
    {
        return Vector3f( float(vec( 0 )), float(vec( 1 )), float(vec( 2 )) );
    };
    // compute coefficient matrix (A) and target matrix (B)
    for ( int k = 0; k < source.size(); k++ )
    {
        const auto& s = source[k];
        const auto& t = target[k];
        auto ws = freeformWeights( pascalLineX,pascalLineY,pascalLineZ,
                                   box.min, backDiagonal, resXY, 
                                   s );
        for ( int i = 0; i < refPointsCout; i++ )
        {
            B.row( i ) += vecToEigen( t - s ) * ws[i];
            for ( int j = 0; j < refPointsCout; j++ )
            {
                A( i, j ) += ws[i] * ws[j];
            }
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> C = A.colPivHouseholderQr().solve( B );

    // Make result equal to origin grid
    std::vector<Vector3f> res = makeOriginGrid( box, resolution );

    // Add calculated diffs to origin grid
    for ( int i = 0; i < refPointsCout; ++i )
    {
        res[i] += eigenToVec( C.row( i ) );
    }
    return res;
}

}
