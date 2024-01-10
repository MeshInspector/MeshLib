#include "MRMeshFwd.h"
#include "MRCone3.h"
#include "MRVector.h"
#include "MRMatrix.h"
#include <Eigen/Eigenvalues>
#include "MRPch/MRTBB.h"
#include "MRToFromEigen.h"
#include "MRConstants.h"
#include <algorithm>


#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 4643) // Forward declaring 'tuple' in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma warning(disable: 4244) // casting float to double 

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#pragma warning(pop)


// Main idea is here: https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf pages 45-51
// Below we will write out the function and Jacobian for minimization by the Levenberg-Marquard method
// and use it to clarify the apex of the cone and the direction of its main axis.

namespace MR
{

// to use Levenberg-Marquardt minimization we need a special type of functor
// to look1: https://github.com/cryos/eigen/blob/master/unsupported/test/NonLinearOptimization.cpp : lmder_functor  
// to look2: https://eigen.tuxfamily.org/dox-devel/unsupported/group__NonLinearOptimization__Module.html  
template <typename T>
struct ConeFittingFunctor
{
    using Scalar = T;
    using InputType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using InputType2 = Eigen::Vector<T, Eigen::Dynamic>;
    using ValueType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using JacobianType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    std::vector <Eigen::Vector3<T>> points;

    int inputs() const
    {
        return 6;
    }
    int values() const
    {
        return static_cast< int > ( points.size() );
    }

    // https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf formula 103 
    // F[i](V,W) = D^T * (I - W * W^T) * D 
    // where: D = V - X[i] and P = (V,W) 
    int operator()( const InputType& x, ValueType& F ) const
    {
        Eigen::Vector3<T> V = x.head<3>();
        Eigen::Vector3<T> W = x.segment<3>( 3 );

        for ( int i = 0; i < points.size(); ++i )
        {
            Eigen::Vector3<T> delta = V - points[i];
            T deltaDotW = delta.dot( W );
            F[i] = delta.squaredNorm() - deltaDotW * deltaDotW;
        }
        return 0;
    }

    // Here we calculate a Jacobian. 
    // function name requested by Eigen lib.
    int df( const InputType& x, JacobianType& J ) const
    {
        Eigen::Vector3<T> V = x.head<3>();
        Eigen::Vector3<T> W = x.segment<3>( 3 ).normalized();

        for ( int i = 0; i < points.size(); ++i )
        {
            const Eigen::Vector3<T>& P = points[i];
            Eigen::Vector3<T> D = ( V - P );
            T PW = D.dot( W );
            Eigen::Vector3<T> PVW = V - P - PW * W;
            Eigen::Vector3<T> PWD = PW * D;

            // Derivative of f with respect to the components of vertex V
            J( i, 0 ) = 2 * PVW.x();
            J( i, 1 ) = 2 * PVW.y();
            J( i, 2 ) = 2 * PVW.z();

            // Derivative of f with respect to the components of the vector W
            J( i, 3 ) = -2 * PWD.x();
            J( i, 4 ) = -2 * PWD.y();
            J( i, 5 ) = -2 * PWD.z();
        }
        return 0;
    }

};


// Class for approximation cloud point by cone. 
// We will calculate the initial approximation of the cone and then use a minimizer to refine the parameters. 
// minimizer is LevenbergMarquardt now. 
// TODO: Possible we could add GaussNewton in future.
template <typename T>
class Cone3Approximation
{
public:
    Cone3Approximation() = default;

    void solve( const std::vector<MR::Vector3<T>>& points,
            Cone3<T>& cone, bool useConeInputAsInitialGuess = false )

    {
        ConeFittingFunctor<T> coneFittingFunctor;
        Eigen::LevenbergMarquardt<ConeFittingFunctor<T>, T> lm( coneFittingFunctor );
        MR::Vector3<T>& coneVertex = cone.apex();
        MR::Vector3<T>& coneAxis = cone.direction();
        T& coneAngle = cone.angle;


        if ( useConeInputAsInitialGuess )
        {
            coneAxis = coneAxis.normalized();
        }
        else
        {
            ComputeInitialCone( points, coneVertex, coneAxis, coneAngle );
        }

        Eigen::VectorX<T> fittedParams( 6 );
        coneToFitParams( cone, fittedParams );
        [[maybe_unused]] Eigen::LevenbergMarquardtSpace::Status result = lm.minimize( fittedParams );

        // Looks like a bug in Eigen. Eigen::LevenbergMarquardtSpace::Status have error codes only. Not return value for Success minimization. 
        // So just log status 

        fitParamsToCone( fittedParams, cone );

        T const one_v = static_cast< T >( 1 );
        auto cosAngle = std::clamp( one_v / coneAxis.length(), static_cast< T >( 0 ), one_v );
        cone.angle = std::acos( cosAngle );
        cone.direction() = cone.direction().normalized();
        cone.length = calculateConeLength( points, cone );

        return;
    }

private:
    T calculateConeLength( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone )
    {
        T length = static_cast< T > ( 0 );
        for ( auto i = 0; i < points.size(); ++i )
        {
            length = std::max( length, std::abs( MR::dot( points[i] - cone.apex(), cone.direction() ) ) );
        }
        return length;
    }

    // Function for finding the best approximation of a straight line in general form y = a*x + b
    void findBestFitLine( const std::vector<MR::Vector2<T>>& xyPairs, T& lineA, T& lineB, MR::Vector2<T>* avg = nullptr )
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
        Eigen::Matrix<T, Eigen::Dynamic, 1> x = A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( b );

        lineA = x( 0 );
        lineB = x( 1 );

        if ( avg )
        {
            *avg = *avg / static_cast < T > ( xyPairs.size() );
            avg->y = lineA * avg->x + lineB;
        }

    }

    void fitParamsToCone( Eigen::Vector<T, Eigen::Dynamic>& fittedParams, Cone3<T>& cone )
    {
        cone.apex().x = fittedParams[0];
        cone.apex().y = fittedParams[1];
        cone.apex().z = fittedParams[2];

        cone.direction().x = fittedParams[3];
        cone.direction().y = fittedParams[4];
        cone.direction().z = fittedParams[5];
    }
    void coneToFitParams( Cone3<T>& cone, Eigen::Vector<T, Eigen::Dynamic>& fittedParams )
    {
        // The fittedParams guess for the cone vertex.
        fittedParams[0] = cone.apex().x;
        fittedParams[1] = cone.apex().y;
        fittedParams[2] = cone.apex().z;

        // The initial guess for the weighted cone axis.
        T coneCosAngle = std::cos( cone.angle );
        fittedParams[3] = cone.direction().x / coneCosAngle;
        fittedParams[4] = cone.direction().y / coneCosAngle;
        fittedParams[5] = cone.direction().z / coneCosAngle;
    }

    void ComputeInitialCone( const std::vector<MR::Vector3<T>>& points, MR::Vector3<T>& coneVertex, MR::Vector3<T>& coneAxis, T& coneAngle )
    {
        // Compute the average of the sample points.
        MR::Vector3<T> center{ 0, 0, 0 };
        for ( auto i = 0; i < points.size(); ++i )
        {
            center += points[i];
        }
        center = center / static_cast< T >( points.size() );

        // The cone axis is estimated from ZZTZ (see the PDF).
        coneAxis = { 0, 0, 0 };
        for ( auto i = 0; i < points.size(); ++i )
        {
            Vector3<T> delta = points[i] - center;
            coneAxis += delta * MR::dot( delta, delta );
        }
        coneAxis = coneAxis.normalized();

        // Compute the signed heights of the points along the cone axis
        // relative to C. These are the projections of the points onto the
        // line C+t*U. Also compute the radial distances of the points
        // from the line C+t*U.
        std::vector<Vector2<T>> hrPairs( points.size() );
        T hMin = std::numeric_limits<T>::max(), hMax = -hMin;
        for ( auto i = 0; i < points.size(); ++i )
        {
            MR::Vector3<T> delta = points[i] - center;
            T h = MR::dot( coneAxis, delta );
            hMin = std::min( hMin, h );
            hMax = std::max( hMax, h );
            Vector3<T> projection = delta - MR::dot( coneAxis, delta ) * coneAxis;
            T r = projection.length();
            hrPairs[i] = { h, r };
        }

        // The radial distance is considered to be a function of height.
        // Fit the (h,r) pairs with a line:
        //   r - rAverage = hrSlope * (h - hAverage)
        MR::Vector2<T> avgPoint;
        T a, b; // line y=a*x+b
        findBestFitLine( hrPairs, a, b, &avgPoint );
        T hAverage = avgPoint.x;
        T rAverage = avgPoint.y;
        T hrSlope = a;

        // If U is directed so that r increases as h increases, U is the
        // correct cone axis estimate. However, if r decreases as h
        // increases, -U is the correct cone axis estimate.
        if ( hrSlope < 0 )
        {
            coneAxis = -coneAxis;
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

        // Using trigonometry and right triangles, compute the tangent
        // function of the cone angle.
        T tanAngle = rRange / hRange;
        coneAngle = std::atan2( rRange, hRange );

        // Compute the cone vertex.
        T offset = rMax / tanAngle - hMax;
        coneVertex = center - offset * coneAxis;
    }
};

}




