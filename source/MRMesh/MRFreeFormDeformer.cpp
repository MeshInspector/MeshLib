#include "MRFreeFormDeformer.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRPch/MRTBB.h"
#include "MRComputeBoundingBox.h"
#include "MRBitSetParallelFor.h"
#include "MRToFromEigen.h"
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
Vector3f interpolateNPoints( const std::span<const Vector3f>& points, float coef, std::vector<Vector3f>& tempPoints )
{
    const float invCoef = 1 - coef;
    if ( points.size() == 2 )
        return points[0] * invCoef + points[1] * coef;

    const size_t N = points.size();

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
double cyclePow( double a, int b )
{
    if ( b == 0 )
        return 1.0f;
    double res = a;
    for ( int i = 1; i < b; ++i )
        res *= a;
    return res;
}

// Optimized by passing precalculated parameters
std::vector<double> freeformWeights( const std::vector<int>& pascalLineX,
                                    const std::vector<int>& pascalLineY,
                                    const std::vector<int>& pascalLineZ,
                                    const Vector3d& boxMin,
                                    const Vector3d& backDiagonal,// 1 - box.diagonal
                                    const size_t resXY,
                                    const Vector3d& point )
{
    Vector3i resolution( int( pascalLineX.size() ), int( pascalLineY.size() ), int( pascalLineZ.size() ) );
    std::vector<double> result( resXY * resolution.z );

    std::vector<double> xCoefs( resolution.x );
    std::vector<double> yCoefs( resolution.y );
    std::vector<double> zCoefs( resolution.z );

    auto minToPoint = point - boxMin;
    auto pointRatio = Vector3d(
        minToPoint.x * backDiagonal.x,
        minToPoint.y * backDiagonal.y,
        minToPoint.z * backDiagonal.z );
    auto backPointRatio = Vector3d::diagonal( 1.0 ) - pointRatio;

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
        size_t index = x + y * resolution.x + z * resXY;
        result[index] = xCoefs[x] * yCoefs[y] * zCoefs[z];
    }
    return result;
}

}

namespace MR
{

std::vector<Vector3f> makeFreeFormOriginGrid( const Box3f& box, const Vector3i& resolution )
{
    auto resXY = resolution.x * resolution.y;
    std::vector<Vector3f> res( resXY * resolution.z );
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

FreeFormDeformer::FreeFormDeformer( VertCoords& coords, const VertBitSet& valid ) :
    coords_{ coords },
    validPoints_{ valid }
{
}

void FreeFormDeformer::init( const Vector3i& resolution /*= Vector3i::diagonal( 2 ) */, const Box3f& initialBox /*= Box3f() */ )
{
    assert( resolution.x > 1 && resolution.y > 1 && resolution.z > 1 );
    initialBox_ = initialBox.valid() ? initialBox : computeBoundingBox( coords_, validPoints_ );
    normedCoords_.resize( coords_.size() );
    auto diagonalVec = initialBox_.max - initialBox_.min;
    diagonalVec.x = 1.0f / diagonalVec.x;
    diagonalVec.y = 1.0f / diagonalVec.y;
    diagonalVec.z = 1.0f / diagonalVec.z;
    BitSetParallelFor( validPoints_, [&] ( VertId vid )
    {
        const auto& point = coords_[vid];
        auto minToPoint = point - initialBox_.min;
        normedCoords_[vid] = Vector3f(
            minToPoint.x * diagonalVec.x,
            minToPoint.y * diagonalVec.y,
            minToPoint.z * diagonalVec.z );
    } );

    resolution_ = resolution;
    refPointsGrid_ = makeFreeFormOriginGrid( initialBox_, resolution );
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
    auto maxRes = std::max( { resolution_.x, resolution_.y, resolution_.z } );

    struct CacheLines
    {
        std::vector<Vector3f> xPlane;
        std::vector<Vector3f> yLine;
        std::vector<Vector3f> buffer;
    };
    tbb::enumerable_thread_specific<CacheLines> caches;

    BitSetParallelFor( validPoints_, [&] ( VertId vid )
    {
        auto& cache = caches.local();
        if ( cache.xPlane.empty() )
            cache.xPlane.resize( resolution_.y * resolution_.z );
        if ( cache.yLine.empty() )
            cache.yLine.resize( resolution_.z );
        if ( cache.buffer.empty() )
            cache.buffer.resize( maxRes * ( maxRes - 1 ) / 2 - 1 );
        coords_[vid] = applyToNormedPoint_( normedCoords_[vid], cache.xPlane, cache.yLine, cache.buffer );
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
    auto maxRes = std::max( { resolution_.x, resolution_.y, resolution_.z } );
    std::vector<Vector3f> buffer(maxRes * (maxRes - 1) / 2 - 1);
    return applyToNormedPoint_( normedPoint, xPlane, yLine, buffer );
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

Vector3f FreeFormDeformer::applyToNormedPoint_( const Vector3f& normedPoint, std::vector<Vector3f>& xPlaneCache, std::vector<Vector3f>& yLineCache, std::vector<Vector3f>& tempPoints ) const
{
    for ( int z = 0; z < resolution_.z; ++z )
        for ( int y = 0; y < resolution_.y; ++y )
        {
            auto index = y + z * resolution_.y;
            auto indexWithX = index * resolution_.x;
            xPlaneCache[index] = interpolateNPoints( std::span<const Vector3f> ( refPointsGrid_.data() + indexWithX, size_t(resolution_.x) ), normedPoint.x, tempPoints );
        }

    for ( int z = 0; z < resolution_.z; ++z )
    {
        yLineCache[z] = interpolateNPoints( std::span<const Vector3f> ( xPlaneCache.data() + z * resolution_.y, size_t( resolution_.y ) ), normedPoint.y, tempPoints );
    }
    return interpolateNPoints( yLineCache, normedPoint.z, tempPoints );
}

std::vector<Vector3f> findBestFreeformDeformation( const Box3f& box, const std::vector<Vector3f>& source, const std::vector<Vector3f>& target, 
                                                   const Vector3i& resolution /*= Vector3i::diagonal( 2 ) */, const AffineXf3f* samplesToBox /*= nullptr */)
{
    FreeFormBestFit ffbf( Box3d( box ), resolution );

    for ( int k = 0; k < source.size(); k++ )
        ffbf.addPair( samplesToBox ? ( *samplesToBox )( source[k] ) : source[k], samplesToBox ? ( *samplesToBox )( target[k] ) : target[k] );

    return ffbf.findBestDeformationReferenceGrid();
}

FreeFormBestFit::FreeFormBestFit( const Box3d& box, const Vector3i& resolution /*= Vector3i::diagonal( 2 ) */ ) :
    box_{ Box3d( box ) },
    resolution_{ resolution }
{
    // This parameters are needed to optimize freeform weights calculations
    resXY_ = size_t( resolution_.x ) * resolution_.y;
    size_ = resXY_ * resolution_.z;
    pascalLineX_ = getPascalTriangleLine( resolution_.x - 1 );
    pascalLineY_ = getPascalTriangleLine( resolution_.y - 1 );
    pascalLineZ_ = getPascalTriangleLine( resolution_.z - 1 );
    reverseDiagonal_ = box_.size();
    reverseDiagonal_.x = 1.0 / reverseDiagonal_.x;
    reverseDiagonal_.y = 1.0 / reverseDiagonal_.y;
    reverseDiagonal_.z = 1.0 / reverseDiagonal_.z;
    accumA_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>( size_, size_ );
    accumB_ = Eigen::Matrix<double, Eigen::Dynamic, 3>( size_, size_t( 3 ) );
    accumA_.setZero();
    accumB_.setZero();
}

void FreeFormBestFit::addPair( const Vector3d& src, const Vector3d& tgt, double w /*= 1.0 */ )
{
    auto ws = freeformWeights( pascalLineX_, pascalLineY_, pascalLineZ_, box_.min, reverseDiagonal_, resXY_, src );
    for ( size_t i = 0; i < size_; ++i )
    {
        accumB_.row( i ) += ( toEigen( tgt - src ) * ws[i] * w );
        for ( size_t j = 0; j < size_; ++j )
        {
            accumA_( i, j ) += ( ws[i] * ws[j] * w );
        }
    }
    sumWeight_ += w;
}

void FreeFormBestFit::addOther( const FreeFormBestFit& other )
{
    if ( other.box_ != box_ || other.resolution_ != resolution_ )
    {
        assert( false && "Only similar instances should be joined" );
        return;
    }
    accumA_ += other.accumA_;
    accumB_ += other.accumB_;

    double sw = sumWeight_ + other.sumWeight_;
    stabilizer_ = ( stabilizer_ * sumWeight_ + other.stabilizer_ * other.sumWeight_ ) / sw;
    sumWeight_ += other.sumWeight_;
}

std::vector<MR::Vector3f> FreeFormBestFit::findBestDeformationReferenceGrid()
{
    stabilize_();

    Eigen::Matrix<double, Eigen::Dynamic, 3> C = accumA_.colPivHouseholderQr().solve( accumB_ );

    // Make result equal to origin grid
    std::vector<Vector3f> res = makeFreeFormOriginGrid( Box3f( box_ ), resolution_ );

    // Add calculated diffs to origin grid
    for ( size_t i = 0; i < size_; ++i )
        res[i] += Vector3f( fromEigen( Eigen::Vector3d( C.row( i ) ) ) );
    return res;
}

void FreeFormBestFit::stabilize_()
{
    if ( stabilizer_ <= 0 )
        return;
    double sw = sumWeight_;
    if ( sw <= 0 )
        sw = 1.0;
    std::vector<Vector3f> refGrid = makeFreeFormOriginGrid( Box3f( box_ ), resolution_ );
    auto w = sw / double( refGrid.size() ) * stabilizer_;
    for ( const auto& refPoint : refGrid )
        addPair( Vector3d( refPoint ), Vector3d( refPoint ), w );
}

}
