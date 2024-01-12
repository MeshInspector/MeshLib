#include "MRConeObject.h"

#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include <Eigen/Dense>
#include "MRConeApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRMeshNormals.h"
#include "MRMeshSubdivide.h"
#include "MRArrow.h"

namespace MR
{

namespace
{
constexpr int cDetailLevel = 64;
constexpr float thicknessArrow = 0.01f;
constexpr float cBaseRadius = 1.0f;
constexpr float cBaseHeight = 1.0f;

constexpr MR::Vector3f base = MR::Vector3f::plusZ();
constexpr MR::Vector3f apex = { 0,0,0 };



float getFeatureRadiusByAngle( float angle )
{
    return cBaseHeight * std::tan( angle );
}
float getAngleByFeatureRadius( float fRadius )
{
    return std::atan( fRadius / cBaseHeight );
}


MR::Matrix3f getRotationMatrix( const Vector3f& normal )
{
    return Matrix3f::rotation( Vector3f::plusZ(), normal );
}


std::shared_ptr<MR::Mesh> makeFeatureCone( int resolution = cDetailLevel )
{

    auto mesh = std::make_shared<MR::Mesh>( makeArrow( base, apex, thicknessArrow, cBaseRadius, cBaseHeight, resolution ) );

    return mesh;
}

}

MR_ADD_CLASS_FACTORY( ConeObject )

Vector3f ConeObject::getDirection() const
{
    return ( xf().A * Vector3f::plusZ() ).normalized();
}

Vector3f ConeObject::getCenter() const
{
    return xf().b;
}

float ConeObject::getLength() const
{
    return xf().A.toScale().z;
}
void ConeObject::setLength( float length )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getNormalyzedFeatueRadius();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( radius * length, radius * length, length );
    setXf( currentXf );
}


float ConeObject::getNormalyzedFeatueRadius( void ) const
{
    return ( xf().A.toScale().x + xf().A.toScale().y ) / 2.0f / getLength();
}
float ConeObject::getAngle() const
{
    return getAngleByFeatureRadius( getNormalyzedFeatueRadius() );
}

void ConeObject::setAngle( float angle )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto featureRedius = getFeatureRadiusByAngle( angle );
    auto length = getLength();
    currentXf.A = getRotationMatrix( direction ) * Matrix3f::scale( featureRedius * length, featureRedius * length, length );
    setXf( currentXf );
}

void ConeObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    currentXf.A = getRotationMatrix( normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
}

void ConeObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

ConeObject::ConeObject()
{
    constructMesh_();
}

ConeObject::ConeObject( const std::vector<Vector3f>& pointsToApprox )
{
    // create mesh
    constructMesh_();

    // calculate cone parameters.
    MR::Cone3<float> result;
    auto fit = Cone3Approximation<float>();
    fit.solve( pointsToApprox, result );

    // setup parameters
    setDirection( result.direction() );
    setCenter( result.center() );
    setAngle( result.angle );
    setLength( result.height );

}

std::shared_ptr<Object> ConeObject::shallowClone() const
{
    auto res = std::make_shared<ConeObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

std::shared_ptr<Object> ConeObject::clone() const
{
    auto res = std::make_shared<ConeObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

void ConeObject::swapBase_( Object& other )
{
    if ( auto coneObject = other.asType<ConeObject>() )
        std::swap( *this, *coneObject );
    else
        assert( false );
}

void ConeObject::serializeFields_( Json::Value& root ) const
{
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( ConeObject::TypeName() );
}

void ConeObject::constructMesh_()
{
    mesh_ = makeFeatureCone();
    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );
    setDirtyFlags( DIRTY_ALL );
}















//////////////////
///// TESTS //////
//////////////////

// 1 LM eigen minimizator test 

#if 0

struct testLMfunctor
{
    // 'm' pairs of (x, f(x))
    Eigen::MatrixXf measuredValues;

    // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
    int operator()( const Eigen::VectorXf& x, Eigen::VectorXf& fvec ) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fvec' has dimensions m x 1
        // It will contain the error for each data point.

        float aParam = x( 0 );
        float bParam = x( 1 );
        float cParam = x( 2 );

        for ( int i = 0; i < values(); i++ )
        {
            float xValue = measuredValues( i, 0 );
            float yValue = measuredValues( i, 1 );

            fvec( i ) = yValue - ( aParam * xValue * xValue + bParam * xValue + cParam );
        }
        return 0;
    }

    // Compute the jacobian of the errors
    int df( const Eigen::VectorXf& x, Eigen::MatrixXf& fjac ) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fjac' has dimensions m x n
        // It will contain the jacobian of the errors, calculated numerically in this case.

        float epsilon;
        epsilon = 1e-5f;

        for ( int i = 0; i < x.size(); i++ )
        {
            Eigen::VectorXf xPlus( x );
            xPlus( i ) += epsilon;
            Eigen::VectorXf xMinus( x );
            xMinus( i ) -= epsilon;

            Eigen::VectorXf fvecPlus( values() );
            operator()( xPlus, fvecPlus );

            Eigen::VectorXf fvecMinus( values() );
            operator()( xMinus, fvecMinus );

            Eigen::VectorXf fvecDiff( values() );
            fvecDiff = ( fvecPlus - fvecMinus ) / ( 2.0f * epsilon );

            fjac.block( 0, i, values(), 1 ) = fvecDiff;
        }

        return 0;
    }

    // Number of data points, i.e. values.
    int m;

    // Returns 'm', the number of values.
    int values() const
    {
        return m;
    }

    // The number of parameters, i.e. inputs.
    int n;

    // Returns 'n', the number of inputs.
    int inputs() const
    {
        return n;
    }

};
#endif 

#if  0

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


// Rosenbrock function (https://habr.com/ru/articles/308626/)
struct RosenbrockFunctor
{
    using InputType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using ValueType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;



    double rosenbrock2( const Eigen::Matrix<double, 2, 1>& x )
    {
        constexpr int rA = 1;
        constexpr int rB = 10;

        auto first = ( rA - x( 0 ) );
        auto second = x( 1 ) - x( 0 ) * x( 0 );
        return first * first + rB * second * second;
    }


    void rosenbrock( const Eigen::Matrix<double, 2, 1>& x, Eigen::Matrix<double, 2, 1>& f )
    {
        double a = 1.0;
        double b = 100.0;
        f( 0 ) = a * ( 1 - x( 0 ) );
        f( 1 ) = b * ( x( 1 ) - x( 0 ) * x( 0 ) );
    };

    // Rosenbrock function jacobian
    void rosenbrockJacobian( const Eigen::Matrix<double, 2, 1>& x, Eigen::Matrix<double, 2, 2>& jacobian )
    {
        double a = 1.0;
        double b = 100.0;
        jacobian( 0, 0 ) = -a;
        jacobian( 0, 1 ) = 0;
        jacobian( 1, 0 ) = 2 * b * x( 0 );
        jacobian( 1, 1 ) = b;
    };
    int operator()( const InputType& x, ValueType& f ) const
    {
        double a = 1.0;
        double b = 100.0;
        f( 0 ) = a * ( 1 - x( 0 ) );
        f( 1 ) = b * ( x( 1 ) - x( 0 ) * x( 0 ) );
        return 0;
    };
    // Compute the jacobian of the errors
    int df( const InputType& x, JacobianType& jacobian ) const
    {
        double a = 1.0;
        double b = 100.0;
        jacobian( 0, 0 ) = -a;
        jacobian( 0, 1 ) = 0;
        jacobian( 1, 0 ) = 2 * b * x( 0 );
        jacobian( 1, 1 ) = b;
        return 0;
    }

};
// GTest test to check the optimization of the Rosenbrock function
TEST( MRMesh, LevenbergMarquardtTestRosenbrockOptimization )
{
    Eigen::VectorX<double> x( 2 );
    x( 0 ) = 2.0;
    x( 1 ) = 2.0;

    // Initialize the LevenbergMarquardt optimization object
    RosenbrockFunctor rosenbrockFunctor;
    Eigen::LevenbergMarquardt<RosenbrockFunctor, double> lm( rosenbrockFunctor );
    lm.parameters.maxfev = 1000; // Maximum number of iterations
    lm.parameters.ftol = 1e-6; // Convergence threshold

    // optimization
    auto result = lm.minimize( x );
    std::cout << "Optimization result: " << result << " x0:" << x( 0 ) << " x1:" << x( 1 ) << std::endl;

    // Checking the optimization result
    EXPECT_NEAR( x( 0 ), 1.0, 1e-3 ); // Проверка значения x1
    EXPECT_NEAR( x( 1 ), 1.0, 1e-3 ); // Проверка значения x2
}
#endif 

}