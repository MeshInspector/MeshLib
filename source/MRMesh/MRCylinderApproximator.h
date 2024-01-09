#include "MRMeshFwd.h"
#include "MRCylinder3.h"
#include "MRVector.h"
#include "MRMatrix.h"
#include <Eigen/Eigenvalues>
#include "MRPch/MRTBB.h"
#include "MRToFromEigen.h"
#include "MRConstants.h"



#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 4643) // Forward declaring 'tuple' in namespace std is not permitted by the C++ Standard.
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma warning(disable: 4244)

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#pragma warning(pop)


// https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf 

namespace MR
{
template <typename T>
class Cylinder3Approximation
{

private:

    enum class CylinderFitterType
    {
        // The algorithm implimentation needs an initial approximation to refine the cylinder axis. 
        // In this option, we sort through several possible options distributed over the hemisphere.
        HemisphereSearchFit,

        // In this case, we assume that there is an external estimate for the cylinder axis. 
        // Therefore, we specify only the position that is given from the outside
        SpecificAxisFit

        // TODO for Meshes try to impliment specific algorithm from https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
        // TODO Also, an estimate of the cylinder axis can be obtained by the gravel component method or the like. But this requires additional. experiments.
        // TODO for future try eigen vector covariance   https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    };

    CylinderFitterType fitter_ = CylinderFitterType::HemisphereSearchFit;

    //  CylinderFitterType::SpecificAxisFit params
    Eigen::Vector<T, 3> baseCylinderAxis_;

    // CylinderFitterType::HemisphereSearchFit params
    size_t thetaResolution_ = 0;
    size_t phiResolution_ = 0;
    bool isMultithread_ = true;

    //Input data converted to Eigen format and normalized to the avgPoint position of all points in the cloud.
    std::vector<Eigen::Vector<T, 3>> normalizedPoints_ = {};

    // Precalculated values for speed up. 
    // In https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf page 35-36: 
    // Text below is a direct copy from pdf file: 
    // The sample application that used equation (94) directly was really slow. 
    // On an Intel CoreTM i7-6700 CPU at 3.40 GHz, the single - threaded version for 10765 points required 129 seconds 
    // and the multithreaded version using 8 hyperthreads required 22 seconds.The evaluation of G using the precomputed summations is much faster.
    // The single - threaded version required 85 milliseconds and the multithreaded version using 8 hyperthreads required 22 milliseconds.

    Eigen::Vector <T, 6>  precomputedMu_ = {};
    Eigen::Matrix <T, 3, 3>  precomputedF0_ = {};
    Eigen::Matrix <T, 3, 6>  precomputedF1_ = {};
    Eigen::Matrix <T, 6, 6>  precomputedF2_ = {};

public:
    Cylinder3Approximation()
    {
        reset();
    };

    void reset()
    {
        thetaResolution_ = 0;
        phiResolution_ = 0;
        precomputedMu_.setZero();
        precomputedF0_.setZero();
        precomputedF1_.setZero();
        precomputedF2_.setZero();
        normalizedPoints_.clear();
    };
    // Solver for CylinderFitterType::HemisphereSearchFit type 
    // thetaResolution_, phiResolution_  must be positive and as large as it posible. Price is CPU time. (~ 100 gives good results).
    T solveGeneral( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder, size_t theta = 180, size_t phi = 90, bool isMultithread = true )
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
    };

    // Solver for CylinderFitterType::SpecificAxisFit type 
    // Simplet way in case of we already know clinder axis 
    T solveSpecificAxis( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder, MR::Vector3<T> const& cylinderAxis )
    {

        baseCylinderAxis_ = MR::toEigen( cylinderAxis.normalized() );
        fitter_ = CylinderFitterType::SpecificAxisFit;
        assert( baseCylinderAxis_.isZero() == false ); //  "The cylinder axis must be nonzero."
        auto result = solve( points, cylinder );
        reset();
        return result;
    };
private:
    // main solver. 
    T solve( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder )
    {
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

    void updatePrecomputeParams( const std::vector <MR::Vector3<T>>& points, Vector3<T>& average )
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

    // Core minimization function.
    // Functional that needs to be minimized to obtain the optimal value of W (i.e. the cylinder axis)
    // General definition is formula 94, but for speed up we use formula 99. 
    // https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf 
    T G( const Eigen::Vector<T, 3>& W, Eigen::Vector<T, 3>& PC, T& rsqr ) const
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
        Eigen::Matrix<T, 3, 3> Q = hatA / trace;
        Eigen::Vector<T, 6> pVec{ P( 0, 0 ), P( 0, 1 ), P( 0, 2 ), P( 1, 1 ), P( 1, 2 ), P( 2, 2 ) };
        Eigen::Vector<T, 3> alpha = precomputedF1_ * pVec;
        Eigen::Vector<T, 3> beta = Q * alpha;
        T error = ( pVec.dot( precomputedF2_ * pVec ) - 4 * alpha.dot( beta ) + 4 * beta.dot( precomputedF0_ * beta ) ) / static_cast< T >( normalizedPoints_.size() );

        PC = beta;
        rsqr = pVec.dot( precomputedMu_ ) + beta.dot( beta );

        return error;
    }

    T fitCylindeHemisphereSingleThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare ) const
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

    class BestHemisphereStoredData
    {
    public:
        T error = std::numeric_limits<T>::max();
        T rootSquare = std::numeric_limits<T>::max();
        Eigen::Vector<T, 3> W;
        Eigen::Vector<T, 3> PC;
    };

    T fitCylindeHemisphereMultiThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare ) const
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
    };

    T SpecificAxisFit( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare )
    {
        W = baseCylinderAxis_;
        return G( W, PC, resultedRootSquare );
    };
};




#if 0 

struct PointCloud
{
    Eigen::MatrixXd points; // Матрица размерности n x 3, где n - количество точек
};


class ConicSurfaceModel
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1> InputType;
    typedef Eigen::Matrix<Scalar, 1, 1> ValueType;
    typedef Eigen::Matrix<Scalar, 1, 3> JacobianType;



    int operator()( const Eigen::VectorXd& x, Eigen::VectorXd& fvec ) const
    {
        const Scalar a = x[0];
        const Scalar b = x[1];
        const Scalar c = x[2];

        for ( int i = 0; i < cloud_.points.rows(); ++i )
        {
            const InputType p = cloud_.points.row( i ).transpose();
            const ValueType residual = ( p.transpose() * A_ * p ) / ( p.transpose() * B_ * p ) - Scalar( 1.0 );
            fvec[i] = residual( 0 );
        }

        return 0;
    }

    int df( const Eigen::VectorXd& x, Eigen::MatrixXd& fjac ) const
    {
        const Scalar a = x[0];
        const Scalar b = x[1];
        const Scalar c = x[2];

        for ( int i = 0; i < cloud_.points.rows(); ++i )
        {
            const InputType p = cloud_.points.row( i ).transpose();
            const ValueType denominator = p.transpose() * B_ * p;
            const JacobianType dfda = 2 * ( B_ * p ) * ( p.transpose() * A_ * p ) / ( denominator * denominator );
            const JacobianType dfdb = -2 * ( B_ * p ) * ( p.transpose() * A_ * p ) / ( denominator * denominator * denominator );
            const JacobianType dfdc = JacobianType::Zero();

            fjac( i, 0 ) = dfda( 0 );
            fjac( i, 1 ) = dfdb( 0 );
            fjac( i, 2 ) = dfdc( 0 );
        }

        return 0;
    }

    PointCloud cloud_;
    InputType A_;
    InputType B_;
};



#endif 



#if 0


struct ConicalFittingFunctor : Functor<double>
{
    ConicalFittingFunctor( const Vector3d& vertex, const Vector3d& axis, double openingAngle, const MatrixXd& points )
        : Functor<double>( points.size() ), vertex_( vertex ), axis_( axis.normalized() ), openingAngle_( openingAngle ), points_( points )
    {}

    int operator()( const VectorXd& x, VectorXd& fvec ) const
    {
        Vector3d apex = vertex_ + x.head( 3 );

        double k = std::tan( openingAngle_ );
        Vector3d dir( x( 3 ), x( 4 ), x( 5 ) );

        for ( int i = 0; i < points_.rows(); ++i )
        {
            const Vector3d& p = points_.row( i );
            Vector3d v = apex - p;
            fvec( i ) = v.dot( axis_ ) - v.norm() * k * dir.dot( v );
        }

        return 0;
    }

private:
    Vector3d vertex_;
    Vector3d axis_;
    double openingAngle_;
    MatrixXd points_;
};

#endif 



template <typename T>
//using T = float;
class Cone3
{
public:
    Cone3()
    {}


    Cone3( const Line3<T>& inAxis, T inAngle, T inLength )
        :
        position( inAxis ),
        angle( inAngle ),
        length( inLength )
    {}

    inline MR::Vector3<T>& center( void )
    {
        return position.p;
    }
    inline MR::Vector3<T>& direction( void )
    {
        return position.d;
    }
    inline MR::Vector3<T>& apex( void )
    {
        return center();
    }


    MR::Line3<T> position; // the combination of the apex of the cone and the direction of its main axis in space
    T angle = 0;
    T length = 0;
};






//using T = float;
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

    // F[i](V,W) = D^T * (I - W * W^T) * D, D = V - X[i], P = (V,W)
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


#if 0 
mFFunction = [this] ( GVector<T> const& P, GVector<T>& F )
{
    Vector3<T> V = { P[0], P[1], P[2] };
    Vector3<T> W = { P[3], P[4], P[5] };
    for ( int32_t i = 0; i < mNumPoints; ++i )
    {
        Vector3<T> delta = V - mPoints[i];
        T deltaDotW = Dot( delta, W );
        F[i] = Dot( delta, delta ) - deltaDotW * deltaDotW;
    }
};
#endif
using T = float;
//template <typename T>
class ApprCone3
{
public:
    ApprCone3()
    {




    }


    // The parameters coneVertex, coneAxis and coneAngle are in/out
    // variables. The caller must provide initial guesses for these.
    // The function estimates the cone parameters and returns them. See
    // GaussNewtonMinimizer.h for a description of the least-squares
    // algorithm and the parameters that it requires. (The file
    // LevenbergMarquardtMinimizer.h directs you to the Gauss-Newton
    // file to read about the parameters.)

            //size_t maxIterations, T updateLengthTolerance, T errorDifferenceTolerance,
//T lambdaFactor, T lambdaAdjust, size_t maxAdjustments,

    void fitParamsToCone( Eigen::Vector<T, Eigen::Dynamic>& fittedParams, Cone3<T>& cone )
    {
        cone.apex().x = fittedParams[0];
        cone.apex().y = fittedParams[1];
        cone.apex().z = fittedParams[2];

        cone.direction().x = fittedParams[3];
        cone.direction().y = fittedParams[4];
        cone.direction().z = fittedParams[5];

        cone.direction().normalized();
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



    void solve( const std::vector<MR::Vector3<T>>& points,
            Cone3<T>& cone  /*, bool useConeInputAsInitialGuess = false */ )

    {
        ConeFittingFunctor<T> coneFittingFunctor;
        Eigen::LevenbergMarquardt<ConeFittingFunctor<T>, T> lm( coneFittingFunctor );
        MR::Vector3<T>  coneVertex, coneAxis;
        T coneAngle;

        /*
        if ( useConeInputAsInitialGuess )
        {
            Normalize( coneAxis );
        }
        else
        */
        {
            ComputeInitialCone( points, coneVertex, coneAxis, coneAngle );
        }

        Eigen::Vector<T, Eigen::Dynamic> fittedParams( 6 );
        coneToFitParams( cone, fittedParams );
        auto result = lm.minimize( fittedParams );
        if ( result == Eigen::LevenbergMarquardtSpace::ImproperInputParameters )
        {
            // LOG ME 
            return;
        }
        else if ( result == Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall )
        {
            // LOG ME 
            return;
        }
        //auto result = minimizer( initial, maxIterations, updateLengthTolerance,
        //    errorDifferenceTolerance, lambdaFactor, lambdaAdjust, maxAdjustments );

        // No test is made for result.converged so that we return some
        // estimates of the cone. The caller can decide how to respond
        // when result.converged is false.

        fitParamsToCone( fittedParams, cone );


        // We know that coneCosAngle will be nonnegative. The std::min
        // call guards against rounding errors leading to a number
        // slightly larger than 1. The clamping ensures std::acos will
        // not return a NaN.
        T const one_v = static_cast< T >( 1 );
        auto coneCosAngle = std::min( one_v / coneAxis.length(), one_v );
        cone.angle = std::acos( coneCosAngle );

        return;
    }



private:


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

    /*
    void findLineApproximation( const std::vector<Vector2<T>>& xyPairs, T& xAvr, T& yAvr, T& a )
    {
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        auto  m = xyPairs.size();

        // Шаг 1: Вычисление средних значений
        xAvr = 0;
        yAvr = 0;
        for ( const auto& pair : xyPairs )
        {
            xAvr += pair.x;
            yAvr += pair.y;
        }
        xAvr /= m;
        yAvr /= m;

        // Создание матрицы A и столбца b
        Matrix A( m, 2 );
        Vector b( m );

        // Шаги 2-4: Заполнение матрицы A и столбца b
        for ( int i = 0; i < m; ++i )
        {
            A( i, 0 ) = xyPairs[i].x - xAvr;
            A( i, 1 ) = 1;
            b( i ) = xyPairs[i].y - yAvr;
        }

        // Шаг 5: Решение линейной системы
        Vector x = A.fullPivLu().solve( b );

        // Шаг 6: Получение значений a и c
        a = x( 0 );
    }
    */


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




