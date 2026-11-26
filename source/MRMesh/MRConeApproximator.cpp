#include "MRConeApproximator.h"
#include "MRBestFit.h"
#include "MRConstants.h"
#include "MRParallelFor.h"
#include "MRVector.h"
#include "MRPch/MRSuppressWarning.h"

#include <Eigen/SVD>

#include <algorithm>

// The initial cone is found as in https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf pages 45-51,
// and then it is refined by refineCone, which minimizes the distances from the points to the cone.

namespace MR
{

namespace
{

/// given cone refined by the points; its height is not updated
template <typename T>
void refine( const Vector<Vector3d>& points, Cone3<T>& cone, size_t maxPoints )
{
    const auto res = refineCone( points, Cone3d( Line3d( Vector3d( cone.apex() ), Vector3d( cone.direction() ) ), cone.angle, cone.height ), maxPoints );
    cone.apex() = Vector3<T>( res.apex() );
    cone.direction() = Vector3<T>( res.direction() ).normalized();
    cone.angle = T( res.angle );
}

} //anonymous namespace

template <typename T>
T Cone3Approximation<T>::solve( const std::vector<MR::Vector3<T>>& points,
    Cone3<T>& cone, const Cone3ApproximationParams& params )
{
    params_ = params;

    switch ( params_.coneFitterType )
    {
    case ConeFitterType::SpecificAxisFit:
        return solveSpecificAxisFit_( points, cone );
        break;
    case ConeFitterType::HemisphereSearchFit:
        return solveHemisphereSearchFit_( points, cone );
        break;
    case ConeFitterType::ApproximationPCM:
        return solveApproximationPCM_( points, cone );
        break;
    default:
        return std::numeric_limits<T>::max();
        break;
    };
}

template <typename T>
T Cone3Approximation<T>::solveFixedAxis_( const std::vector<MR::Vector3<T>>& points,
    Cone3<T>& cone, bool useInputAxis )
{
    MR::Vector3<T> center, U;
    computeCenterAndNormal_( points, center, U );
    // only the axis of the input cone is used
    cone = computeInitialCone_( points, center, useInputAxis ? cone.direction().normalized() : U );
    refine( Vector<Vector3d>( points.begin(), points.end() ), cone, params_.maxRefinePoints );
    cone.height = calculateConeHeight_( points, cone );

    return getApproximationRMS_( points, cone );
}

template <typename T>
T Cone3Approximation<T>::solveApproximationPCM_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone )
{
    return solveFixedAxis_( points, cone, false );
}

template <typename T>
T Cone3Approximation<T>::solveSpecificAxisFit_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone )
{
    return solveFixedAxis_( points, cone, true );
}

template <typename T>
T Cone3Approximation<T>::solveHemisphereSearchFit_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone )
{
    Vector3<T> center = computeCenter_( points );
    const Vector<Vector3d> pointsDouble( points.begin(), points.end() );

    constexpr T pi2 = static_cast< T >( PI2 );
    T const theraStep = static_cast< T >( 2 * PI ) / params_.hemisphereSearchThetaResolution;
    T const phiStep = pi2 / params_.hemisphereSearchPhiResolution;

    struct BestCone {
        Cone3<T> bestCone;
        T minError = std::numeric_limits<T> ::max();
    };
    std::vector<BestCone> bestCones;
    bestCones.resize( params_.hemisphereSearchPhiResolution + 1 );

    ParallelFor( 0, params_.hemisphereSearchPhiResolution + 1, [&] ( int j )
    {
        T phi = phiStep * j; //  [0 .. pi/2]
        T cosPhi = std::cos( phi );
        T sinPhi = std::sin( phi );
        for ( size_t i = 0; i < params_.hemisphereSearchThetaResolution; ++i )
        {
            T theta = theraStep * i; //  [0 .. 2*pi)
            T cosTheta = std::cos( theta );
            T sinTheta = std::sin( theta );

            // cone main axis original extimation
            Vector3<T> U( cosTheta * sinPhi, sinTheta * sinPhi, cosPhi );

            auto tmpCone = computeInitialCone_( points, center, U );

            // calculate approximation error and store best result.
            T error = getApproximationRMS_( points, tmpCone );
            if ( error < bestCones[j].minError )
            {
                bestCones[j].minError = error;
                bestCones[j].bestCone = tmpCone;
            }
        }
    } );

    // only the best initial cone of each row of directions is refined
    ParallelFor( size_t( 0 ), bestCones.size(), [&] ( size_t i )
    {
        auto& best = bestCones[i];
        if ( best.minError == std::numeric_limits<T>::max() )
            return;
        auto refined = best.bestCone;
        refine( pointsDouble, refined, params_.maxRefinePoints );
        if ( const T error = getApproximationRMS_( points, refined ); error < best.minError )
        {
            best.minError = error;
            best.bestCone = refined;
        }
    } );
    auto bestAppox = std::min_element( bestCones.begin(), bestCones.end(), [] ( const BestCone& a, const BestCone& b )
    {
        return a.minError < b.minError;
    } );

    cone = bestAppox->bestCone;

    // calculate cone height
    cone.height = calculateConeHeight_( points, cone );

    return bestAppox->minError;
}

template <typename T>
T Cone3Approximation<T>::calculateConeHeight_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone )
{
    T length = static_cast< T > ( 0 );
    for ( auto i = 0; i < points.size(); ++i )
    {
        length = std::max( length, std::abs( MR::dot( points[i] - cone.apex(), cone.direction() ) ) );
    }
    return length;
}

template <typename T>
T Cone3Approximation<T>::getApproximationRMS_( const std::vector<MR::Vector3<T>>& points, const Cone3<T>& cone )
{
    if ( points.size() == 0 )
        return std::numeric_limits<T>::max();

    T error = 0;
    for ( auto p : points )
        error = error + distanceSq( cone.projectPoint( p ), p );

    return error / points.size();
}

template <typename T>
MR::Vector3<T> Cone3Approximation<T>::computeCenter_( const std::vector<MR::Vector3<T>>& points )
{
    // Compute the average of the sample points.
    MR::Vector3<T> center;  // C in pdf
    for ( auto i = 0; i < points.size(); ++i )
    {
        center += points[i];
    }
    center = center / static_cast< T >( points.size() );
    return center;
}

template <typename T>
void Cone3Approximation<T>::computeCenterAndNormal_( const std::vector<MR::Vector3<T>>& points, MR::Vector3<T>& center, MR::Vector3<T>& U )
{
    center = computeCenter_( points );

    // The cone axis is estimated from ZZTZ (see the https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf, formula 120).
    U = Vector3f();  // U in pdf
    for ( auto i = 0; i < points.size(); ++i )
    {
        Vector3<T> Z = points[i] - center;
        U += Z * MR::dot( Z, Z );
    }
    U = U.normalized();
}

template <typename T>
Cone3<T> Cone3Approximation<T>::computeInitialCone_( const std::vector<MR::Vector3<T>>& points, const MR::Vector3<T>& center, const MR::Vector3<T>& axis )
{
    Cone3<T> result;
    MR::Vector3<T>& coneApex = result.apex();
    result.direction() = axis;
    MR::Vector3<T>& U = result.direction();  // coneAxis
    T& coneAngle = result.angle;

    // C is center, U is coneAxis, X is points
    // Compute the signed heights of the points along the cone axis relative to C.
    // These are the projections of the points onto the line C+t*U. Also compute
    // the radial distances of the points from the line C+t*U

    std::vector<Vector2<T>> hrPairs( points.size() );
    T hMin = std::numeric_limits<T>::max(), hMax = -hMin;
    for ( auto i = 0; i < points.size(); ++i )
    {
        MR::Vector3<T> delta = points[i] - center;
        T h = MR::dot( U, delta );
        hMin = std::min( hMin, h );
        hMax = std::max( hMax, h );
        Vector3<T> projection = delta - MR::dot( U, delta ) * U;
        T r = projection.length();
        hrPairs[i] = { h, r };
    }

    // The radial distance is considered to be a function of height. Fit the
    // (h,r) pairs with a line: r = rAverage = hrSlope * (h = hAverage);

    MR::Vector2<T> avgPoint;
    T a, b; // line y=a*x+b
    findBestFitLine_( hrPairs, a, b, &avgPoint );
    T hAverage = avgPoint.x;
    T rAverage = avgPoint.y;
    T hrSlope = a;

    // If U is directed so that r increases as h increases, U is the correct
    // cone axis estimate. However, if r decreases as h increases, -U is the
    // correct cone axis estimate.
    if ( hrSlope < 0 )
    {
        U = -U;
        hrSlope = -hrSlope;
        std::swap( hMin, hMax );
        hMin = -hMin;
        hMax = -hMax;
    }

    // Compute the extreme radial distance values for the points.
    T rMin = rAverage + hrSlope * ( hMin - hAverage );
    T rMax = rAverage + hrSlope * ( hMax - hAverage );
    T hRange = hMax - hMin;
    T rRange = rMax - rMin;

    // Using trigonometry and right triangles, compute the tangent function of the cone angle.
    T tanAngle = rRange / hRange;
    coneAngle = std::atan2( rRange, hRange );

    // Compute the cone vertex.
    T offset = rMax / tanAngle - hMax;
    coneApex = center - offset * U;
    return result;
}

template <typename T>
void Cone3Approximation<T>::findBestFitLine_( const std::vector<MR::Vector2<T>>& xyPairs, T& lineA, T& lineB, MR::Vector2<T>* avg )
{
    auto numPoints = xyPairs.size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A( numPoints, 2 );
    Eigen::Vector<T, Eigen::Dynamic>  b( numPoints );

    for ( auto i = 0; i < numPoints; ++i )
    {
        A( i, 0 ) = xyPairs[i].x;  // x-coordinate of the i-th point
        A( i, 1 ) = 1.0;           // constant 1.0 for dummy term b
        b( i ) = xyPairs[i].y;     // y-coordinate of the i-th point
        if ( avg )
            *avg = *avg + xyPairs[i];
    }
    if ( avg )
    {
        *avg = *avg / static_cast < T > ( xyPairs.size() );
    }
    // Solve the system of equations Ax = b using the least squares method
    MR_SUPPRESS_WARNING_PUSH
    MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )
    Eigen::Matrix<T, Eigen::Dynamic, 1> x = A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( b );
    MR_SUPPRESS_WARNING_POP

    lineA = x( 0 );
    lineB = x( 1 );

    if ( avg )
    {
        *avg = *avg / static_cast < T > ( xyPairs.size() );
        avg->y = lineA * avg->x + lineB;
    }
}

template class Cone3Approximation<float>;

}
