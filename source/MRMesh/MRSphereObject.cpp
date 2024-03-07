#include "MRSphereObject.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRMesh/MRDefaultFeatureObjectParams.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3Decompose.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:5054)  //operator '&': deprecated between enumerations of different types
#pragma warning(disable:4127)  //C4127. "Consider using 'if constexpr' statement instead"
#elif defined(__clang__)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <Eigen/Dense>

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__clang__)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


namespace MR
{

MR_ADD_CLASS_FACTORY( SphereObject )

float SphereObject::getRadius() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return s.x.x;
}

Vector3f SphereObject::getCenter() const
{
    return xf().b;
}

void SphereObject::setRadius( float radius )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::scale( radius );
    setXf( currentXf );
}

void SphereObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}


const std::vector<FeatureObjectSharedProperty>& SphereObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Radius", &SphereObject::getRadius, &SphereObject::setRadius},
       {"Center", &SphereObject::getCenter, &SphereObject::setCenter}
    };
    return ret;
}

SphereObject::SphereObject()
{
    setDefaultFeatureObjectParams( *this );
}

SphereObject::SphereObject( const std::vector<Vector3f>& pointsToApprox )
    : SphereObject()
{
    // find best radius and center
    Eigen::Matrix<double, 4, 4> accumA_;
    Eigen::Matrix<double, 4, 1> accumB_;
    accumA_.setZero();
    accumB_.setZero();
    for ( const auto& pta : pointsToApprox )
    {
        Eigen::Matrix<double, 4, 1> vec;
        vec[0] = 2.0 * pta.x;
        vec[1] = 2.0 * pta.y;
        vec[2] = 2.0 * pta.z;
        vec[3] = -1.0;

        accumA_ += ( vec * vec.transpose() );
        accumB_ += ( vec * ( pta.x * pta.x + pta.y * pta.y + pta.z * pta.z ) );
    }
    Eigen::Matrix<double, 4, 1> res = Eigen::Matrix<double, 4, 1>( accumA_.colPivHouseholderQr().solve( accumB_ ) );
    setCenter( { float( res[0] ),float( res[1] ),float( res[2] ) } );
    double rSq = res[0] * res[0] + res[1] * res[1] + res[2] * res[2] - res[3];
    assert( rSq >= 0.0 );
    setRadius( float( sqrt( std::max( rSq, 0.0 ) ) ) );
}

std::shared_ptr<Object> SphereObject::shallowClone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> SphereObject::clone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

void SphereObject::swapBase_( Object& other )
{
    if ( auto sphereObject = other.asType<SphereObject>() )
        std::swap( *this, *sphereObject );
    else
        assert( false );
}

void SphereObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( SphereObject::TypeName() );
}

void SphereObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype(*this)>( *this );
}

MeasurementPropertyParameters<RadiusVisualizePropertyType> SphereObject::getMeasurementParametersFor_( RadiusVisualizePropertyType index ) const
{
    (void)index; // Only one measurement.
    return {
        .vis = {
            .drawAsDiameter = true,
            .isSpherical = true,
        },
    };
}

}
