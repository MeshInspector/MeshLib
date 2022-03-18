#include "MRSphereObject.h"
#include "MRMatrix3.h"
#include "MRSphere.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include <Eigen/Dense>

namespace
{
constexpr int cDetailLevel = 2048;
constexpr float cBaseRadius = 1.0f;
}

namespace MR
{

MR_ADD_CLASS_FACTORY( SphereObject )

float SphereObject::getRadius() const
{
    return xf().A.toScale().x;
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

SphereObject::SphereObject()
{
    constructMesh_();
}

SphereObject::SphereObject( const std::vector<Vector3f>& pointsToApprox )
{
    constructMesh_();
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

std::vector<std::string> SphereObject::getInfoLines() const
{
    std::vector<std::string> res;

    res.push_back( "type : SphereObject" );
    return res;
}

std::shared_ptr<Object> SphereObject::shallowClone() const
{
    auto res = std::make_shared<SphereObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

std::shared_ptr<Object> SphereObject::clone() const
{
    auto res = std::make_shared<SphereObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
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
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( SphereObject::TypeName() );
}

void SphereObject::constructMesh_()
{
    mesh_ = std::make_shared<Mesh>( makeSphere( { cBaseRadius,cDetailLevel } ) );
    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );
    setDirtyFlags( DIRTY_ALL );
}

}