#include "MRCylinderApproximator.h"
#include "MRConstants.h"
#include "MRCylinder3.h"
#include "MRToFromEigen.h"
#include "MRVector.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

namespace MR
{

template <typename T>
Cylinder3Approximation<T>::Cylinder3Approximation()
{
    reset();
}

template <typename T>
void Cylinder3Approximation<T>::reset()
{
    thetaResolution_ = 0;
    phiResolution_ = 0;
    precomputedMu_.setZero();
    precomputedF0_.setZero();
    precomputedF1_.setZero();
    precomputedF2_.setZero();
    normalizedPoints_.clear();
}

template <typename T>
T Cylinder3Approximation<T>::solveGeneral( const std::vector<Vector3<T>>& points, Cylinder3<T>& cylinder, size_t theta,
    size_t phi, bool isMultithread )
{
    thetaResolution_ = theta;
    phiResolution_ = phi;
    isMultithread_ = isMultithread;
    fitter_ = CylinderFitterType::HemisphereSearchFit;
    assert( thetaResolution_ > 0 );
    assert( phiResolution_ > 0 );
    auto result = solve( points, cylinder );
    reset();
    return result;
}

template <typename T>
T Cylinder3Approximation<T>::solveSpecificAxis( const std::vector<Vector3<T>>& points, Cylinder3<T>& cylinder,
    Vector3<T> const& cylinderAxis )
{
    baseCylinderAxis_ = MR::toEigen( cylinderAxis.normalized() );
    fitter_ = CylinderFitterType::SpecificAxisFit;
    assert( baseCylinderAxis_.isZero() == false ); //  "The cylinder axis must be nonzero."
    auto result = solve( points, cylinder );
    reset();
    return result;
}

template <typename T>
T Cylinder3Approximation<T>::solve( const std::vector<Vector3<T>>& points, Cylinder3<T>& cylinder )
{
    if ( points.size() < 6 )
    {
        spdlog::warn( "Cylinder3Approximation :: Too low point for cylinder approximation count=" + std::to_string( points.size() ) );
        return -1;
    }
    assert( points.size() >= 6 ); // "At least 6 points needs for correct fitting requires ."

    normalizedPoints_.clear();
    cylinder = Cylinder3<T>();
    Vector3<T> avgPoint;
    Eigen::Vector<T, 3> bestPC;
    Eigen::Vector<T, 3> bestW; // cylinder main axis
    T rootSquare = 0;
    T error = 0;

    //For significant increase speed, we make a preliminary calculation based on the initial data.
    updatePrecomputeParams( points, avgPoint );

    // Listing 16. page 38.
    if ( fitter_ == CylinderFitterType::HemisphereSearchFit )
    {
        if ( isMultithread_ )
            error = fitCylindeHemisphereMultiThreaded( bestPC, bestW, rootSquare );

        else
            error = fitCylindeHemisphereSingleThreaded( bestPC, bestW, rootSquare );
    }
    else if ( fitter_ == CylinderFitterType::SpecificAxisFit )
    {
        error = SpecificAxisFit( bestPC, bestW, rootSquare );
    }
    else
    {
        spdlog::warn( "Cylinder3Approximation :: unsupported fitter" );
        assert( false ); // unsupported fitter
        return -1;
    }

    assert( rootSquare >= 0 );

    cylinder.center() = fromEigen( bestPC ) + avgPoint;
    cylinder.direction() = ( fromEigen( bestW ) ).normalized();
    cylinder.radius = std::sqrt( rootSquare );

    // Calculate a max. possible length of a cylinder covered by dataset.
    // Project point on a main cylinder axis
    T hmin = std::numeric_limits<T>::max();
    T hmax = -std::numeric_limits<T>::max();

    for ( size_t i = 0; i < points.size(); ++i )
    {
        T h = MR::dot( cylinder.direction(), points[i] - cylinder.center() );
        hmin = std::min( h, hmin );
        hmax = std::max( h, hmax );
    }
    T hmid = ( hmin + hmax ) / 2;

    // Very tiny correct a cylinder center.
    cylinder.center() = cylinder.center() + hmid * cylinder.direction();
    cylinder.length = hmax - hmin;

    assert( cylinder.length >= 0 );

    return error;
}

template <typename T>
void Cylinder3Approximation<T>::updatePrecomputeParams( const std::vector<Vector3<T>>& points, Vector3<T>& average )
{
    // Listing 15. page 37.
    normalizedPoints_.resize( points.size() );

    // calculate avg point of dataset
    average = Vector3<T>{};
    for ( size_t i = 0; i < points.size(); ++i )
        average += points[i];
    average = average / static_cast< T > ( points.size() );

    // normalize points and store it.
    for ( size_t i = 0; i < points.size(); ++i )
        normalizedPoints_[i] = toEigen( points[i] - average );

    const Eigen::Vector<T, 6> zeroEigenVector6{ 0, 0, 0, 0, 0, 0 };
    std::vector<Eigen::Vector<T, 6>> products( normalizedPoints_.size(), zeroEigenVector6 );
    precomputedMu_ = zeroEigenVector6;

    for ( size_t i = 0; i < normalizedPoints_.size(); ++i )
    {
        products[i][0] = normalizedPoints_[i][0] * normalizedPoints_[i][0];
        products[i][1] = normalizedPoints_[i][0] * normalizedPoints_[i][1];
        products[i][2] = normalizedPoints_[i][0] * normalizedPoints_[i][2];
        products[i][3] = normalizedPoints_[i][1] * normalizedPoints_[i][1];
        products[i][4] = normalizedPoints_[i][1] * normalizedPoints_[i][2];
        products[i][5] = normalizedPoints_[i][2] * normalizedPoints_[i][2];
        precomputedMu_[0] += products[i][0];
        precomputedMu_[1] += 2 * products[i][1];
        precomputedMu_[2] += 2 * products[i][2];
        precomputedMu_[3] += products[i][3];
        precomputedMu_[4] += 2 * products[i][4];
        precomputedMu_[5] += products[i][5];
    }
    precomputedMu_ = precomputedMu_ / points.size();

    precomputedF0_.setZero();
    precomputedF1_.setZero();
    precomputedF2_.setZero();
    for ( size_t i = 0; i < normalizedPoints_.size(); ++i )
    {
        Eigen::Vector<T, 6> delta{};
        delta[0] = products[i][0] - precomputedMu_[0];
        delta[1] = 2 * products[i][1] - precomputedMu_[1];
        delta[2] = 2 * products[i][2] - precomputedMu_[2];
        delta[3] = products[i][3] - precomputedMu_[3];
        delta[4] = 2 * products[i][4] - precomputedMu_[4];
        delta[5] = products[i][5] - precomputedMu_[5];
        precomputedF0_( 0, 0 ) += products[i][0];
        precomputedF0_( 0, 1 ) += products[i][1];
        precomputedF0_( 0, 2 ) += products[i][2];
        precomputedF0_( 1, 1 ) += products[i][3];
        precomputedF0_( 1, 2 ) += products[i][4];
        precomputedF0_( 2, 2 ) += products[i][5];
        precomputedF1_ = precomputedF1_ + normalizedPoints_[i] * delta.transpose();
        precomputedF2_ += delta * delta.transpose();
    }
    precomputedF0_ = precomputedF0_ / static_cast < T > ( points.size() );
    precomputedF0_( 1, 0 ) = precomputedF0_( 0, 1 );
    precomputedF0_( 2, 0 ) = precomputedF0_( 0, 2 );
    precomputedF0_( 2, 1 ) = precomputedF0_( 1, 2 );
    precomputedF1_ = precomputedF1_ / static_cast < T > ( points.size() );
    precomputedF2_ = precomputedF2_ / static_cast < T > ( points.size() );
}

template <typename T>
T Cylinder3Approximation<T>::G( const Eigen::Vector<T, 3>& W, Eigen::Vector<T, 3>& PC, T& rsqr ) const
{
    Eigen::Matrix<T, 3, 3> P = Eigen::Matrix<T, 3, 3>::Identity() - ( W * W.transpose() ); // P = I - W * W^T

    Eigen::Matrix<T, 3, 3> S;
    S << 0, -W[2], W[1],
        W[2], 0, -W[0],
        -W[1], W[0], 0;

    Eigen::Matrix<T, 3, 3> A = P * precomputedF0_ * P;
    Eigen::Matrix<T, 3, 3> hatA = -( S * A * S );
    Eigen::Matrix<T, 3, 3> hatAA = hatA * A;
    T trace = hatAA.trace();
    if ( trace == 0 )
    {
        // cannot divide on zero, return maximum error
        PC.setZero();
        return std::numeric_limits<T>::max();
    }
    Eigen::Matrix<T, 3, 3> Q = hatA / trace;
    Eigen::Vector<T, 6> pVec{ P( 0, 0 ), P( 0, 1 ), P( 0, 2 ), P( 1, 1 ), P( 1, 2 ), P( 2, 2 ) };
    Eigen::Vector<T, 3> alpha = precomputedF1_ * pVec;
    Eigen::Vector<T, 3> beta = Q * alpha;
    T error = ( pVec.dot( precomputedF2_ * pVec ) - 4 * alpha.dot( beta ) + 4 * beta.dot( precomputedF0_ * beta ) ) / static_cast< T >( normalizedPoints_.size() );

    // some times appears floating points calculation errors. Error is a non negative value by default, so, make it positive.
    // absolute value (instead of error=0) is used to avoid collisions for near-null values and subsequent ambiguous work, since this code can be used in parallel algorithms
    if ( error < 0 )
        error = std::abs( error );

    PC = beta;
    rsqr = std::max( T(0), pVec.dot( precomputedMu_ ) + beta.dot( beta ) ); // the value can slightly below zero due to rounding-errors

    return error;
}

template <typename T>
T Cylinder3Approximation<T>::fitCylindeHemisphereSingleThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W,
    T& resultedRootSquare ) const
{
    T const theraStep = static_cast< T >( 2 * PI ) / thetaResolution_;
    T const phiStep = static_cast< T >( PI2 ) / phiResolution_;

    // problem in list. 16. => start from pole.
    W = { 0, 0, 1 };
    T minError = G( W, PC, resultedRootSquare );

    for ( size_t j = 1; j <= phiResolution_; ++j )
    {
        T phi = phiStep * j; //  [0 .. pi/2]
        T cosPhi = std::cos( phi );
        T sinPhi = std::sin( phi );
        for ( size_t i = 0; i < thetaResolution_; ++i )
        {
            T theta = theraStep * i; //  [0 .. 2*pi)
            T cosTheta = std::cos( theta );
            T sinTheta = std::sin( theta );
            Eigen::Vector<T, 3> currW{ cosTheta * sinPhi, sinTheta * sinPhi, cosPhi };
            Eigen::Vector<T, 3> currPC{};
            T rsqr;
            T error = G( currW, currPC, rsqr );
            if ( error < minError )
            {
                minError = error;
                resultedRootSquare = rsqr;
                W = currW;
                PC = currPC;
            }
        }
    }

    return minError;
}

template <typename T>
T Cylinder3Approximation<T>::fitCylindeHemisphereMultiThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W,
    T& resultedRootSquare ) const
{
    T const theraStep = static_cast< T >( 2 * PI ) / thetaResolution_;
    T const phiStep = static_cast< T >( PI2 ) / phiResolution_;

    // problem in list. 16. => start from pole.
    W = { 0, 0, 1 };
    T minError = G( W, PC, resultedRootSquare );

    std::vector<BestHemisphereStoredData> storedData;
    storedData.resize( phiResolution_ + 1 ); //  [0 .. pi/2] +1 for include upper bound

    tbb::parallel_for( tbb::blocked_range<size_t>( size_t( 0 ), phiResolution_ + 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t j = range.begin(); j < range.end(); ++j )
            {

                T phi = phiStep * j; // [0 .. pi/2]
                T cosPhi = std::cos( phi );
                T sinPhi = std::sin( phi );
                for ( size_t i = 0; i < thetaResolution_; ++i )
                {

                    T theta = theraStep * i; // [0 .. 2*pi)
                    T cosTheta = std::cos( theta );
                    T sinTheta = std::sin( theta );
                    Eigen::Vector<T, 3> currW{ cosTheta * sinPhi, sinTheta * sinPhi, cosPhi };
                    Eigen::Vector<T, 3> currPC{};
                    T rsqr;
                    T error = G( currW, currPC, rsqr );

                    if ( error < storedData[j].error )
                    {
                        storedData[j].error = error;
                        storedData[j].rootSquare = rsqr;
                        storedData[j].W = currW;
                        storedData[j].PC = currPC;
                    }
                }
            }
        }
    );

    for ( size_t i = 0; i <= phiResolution_; ++i )
    {
        if ( storedData[i].error < minError )
        {
            minError = storedData[i].error;
            resultedRootSquare = storedData[i].rootSquare;
            W = storedData[i].W;
            PC = storedData[i].PC;
        }
    }

    return minError;
}

template <typename T>
T Cylinder3Approximation<T>::SpecificAxisFit( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare )
{
    W = baseCylinderAxis_;
    return G( W, PC, resultedRootSquare );
}

template class Cylinder3Approximation<float>;
template class Cylinder3Approximation<double>;

} // namespace MR
