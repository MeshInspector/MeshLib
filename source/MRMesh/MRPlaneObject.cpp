#include "MRPlaneObject.h"
#include "MRMesh.h"
#include "MRMesh/MRDefaultFeatureObjectParams.h"
#include "MRMeshBuilder.h"
#include "MRBestFit.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3.h"
#include "MRVector3.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

// Offset in positive and negative directions along the X and Y axes when constructing a base object.
// Historically it eq. 1,  which means that original plane have a 2x2 size.
// basePlaneObjectHalfEdgeLength_=0.5 looks better.
// But left as is for compatibility.
constexpr float basePlaneObjectHalfEdgeLength_ = 1.0f;

MR_ADD_CLASS_FACTORY( PlaneObject )

Vector3f PlaneObject::getNormal() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return ( r * Vector3f::plusZ() ).normalized();
}

Vector3f PlaneObject::getCenter() const
{
    return xf().b;
}

void PlaneObject::setNormal( const Vector3f& normal )
{
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * s;
    setXf( currentXf );
    orientateFollowMainAxis_();

}

void PlaneObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

void PlaneObject::setSize( float size )
{
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = r * Matrix3f::scale( Vector3f::diagonal( size / basePlaneObjectHalfEdgeLength_ / 2.0f ) );
    setXf( currentXf );
}

/*
void CylinderObject::setLength( float length )
{
    auto direction = getDirection();
    auto currentXf = xf();
    auto radius = getRadius();
    currentXf.A = ( getRotationMatrix( direction ) * Matrix3f::scale( radius, radius, length ) );
    setXf( currentXf );
}
*/

void PlaneObject::setSizeX( float size )
{
    size = size / basePlaneObjectHalfEdgeLength_ / 2.0f; // normalization for base figure dimentions
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = r * Matrix3f::scale( size, s.y.y, 1.0f );
    setXf( currentXf );
}

void PlaneObject::setSizeY( float size )
{
    size = size / basePlaneObjectHalfEdgeLength_ / 2.0f; // normalization for base figure dimentions
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = r * Matrix3f::scale( s.x.x, size, 1.0f );
    setXf( currentXf );
}


float PlaneObject::getSize( void ) const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return  s.x.x * basePlaneObjectHalfEdgeLength_ * 2.0f;
}

float PlaneObject::getSizeX( void ) const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return  s.x.x * basePlaneObjectHalfEdgeLength_ * 2.0f;
}

float PlaneObject::getSizeY( void ) const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return  s.y.y * basePlaneObjectHalfEdgeLength_ * 2.0f;
}

const std::vector<FeatureObjectSharedProperty>& PlaneObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
        {"Center", &PlaneObject::getCenter,&PlaneObject::setCenter},
        {"Normal", &PlaneObject::getNormal,&PlaneObject::setNormal},
        {"Size"  , &PlaneObject::getSize,  &PlaneObject::setSize  }
    };
    return ret;
}

PlaneObject::PlaneObject()
{
    setDefaultFeatureObjectParams( *this );
    constructMesh_();
}


void PlaneObject::orientateFollowMainAxis_()
{
    auto axis = Vector3f::plusZ();
    auto vector = cross( axis, getNormal() );

    constexpr float parallelVectorLimitSq = 1e-4f;
    if ( vector.lengthSq() < parallelVectorLimitSq )
        vector = Vector3f( { axis.y , axis.z , axis.x } );
    vector = vector.normalized();

    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    auto x = ( r * MR::Vector3f::plusX() ).normalized();

    auto angle = std::atan2( cross( x, vector ).length(), dot( x, vector ) );
    auto A = Matrix3f::rotation( MR::Vector3f::plusZ(), angle );
    angle = angle * 180.0f / PI_F;

    auto currXf = xf();
    currXf.A = r * A * s;
    setXf( currXf );
}

PlaneObject::PlaneObject( const std::vector<Vector3f>& pointsToApprox )
    : PlaneObject()
{
    PointAccumulator pa;
    Box3f box;
    for ( const auto& p : pointsToApprox )
    {
        pa.addPoint( p );
        box.include( p );
    }

    // make a normal vector from center directed against a point (0, 0, 0)
    Plane3f plane = pa.getBestPlanef();
    Vector3f normal = plane.n.normalized();
    if ( plane.d < 0 )
        normal *= -1.f;

    setNormal( normal );

    setCenter( plane.project( box.center() ) );
    //setSize( box.diagonal() );

    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );



    MR::Vector3f min( std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() );
    MR::Vector3f max( -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() );

    auto oX = ( r * Vector3f::plusX() ).normalized();
    auto oY = ( r * Vector3f::plusY() ).normalized();
    auto oZ = ( r * Vector3f::plusZ() ).normalized();



    for ( const auto& p : pointsToApprox )
    {
        auto dX = MR::dot( oX, p );
        if ( dX < min.x )
            min.x = dX;
        if ( dX > max.x )
            max.x = dX;

        auto dY = MR::dot( oY, p );
        if ( dY < min.y )
            min.y = dY;
        if ( dY > max.y )
            max.y = dY;

        auto dZ = MR::dot( oZ, p );
        if ( dZ < min.z )
            min.z = dZ;
        if ( dZ > max.z )
            max.z = dZ;

    }

    auto sX = std::abs( max.x - min.x );
    auto sY = std::abs( max.y - min.y );
    auto sZ = std::abs( max.z - min.z );

    sZ = sZ;
    setSizeX( sX );
    setSizeY( sY );
}

std::shared_ptr<Object> PlaneObject::shallowClone() const
{
    auto res = std::make_shared<PlaneObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

std::shared_ptr<Object> PlaneObject::clone() const
{
    auto res = std::make_shared<PlaneObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

void PlaneObject::swapBase_( Object& other )
{
    if ( auto planeObject = other.asType<PlaneObject>() )
        std::swap( *this, *planeObject );
    else
        assert( false );
}

void PlaneObject::serializeFields_( Json::Value& root ) const
{
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( PlaneObject::TypeName() );
}

void PlaneObject::constructMesh_()
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 2_v, 1_v, 3_v }
    };

    // create object Mesh cube
    Mesh meshObj;
    meshObj.topology = MeshBuilder::fromTriangles( t );
    meshObj.points.emplace_back( -basePlaneObjectHalfEdgeLength_, -basePlaneObjectHalfEdgeLength_, 0 ); // VertId{0}
    meshObj.points.emplace_back( basePlaneObjectHalfEdgeLength_, -basePlaneObjectHalfEdgeLength_, 0 ); // VertId{1}
    meshObj.points.emplace_back( -basePlaneObjectHalfEdgeLength_, basePlaneObjectHalfEdgeLength_, 0 ); // VertId{2}
    meshObj.points.emplace_back( basePlaneObjectHalfEdgeLength_, basePlaneObjectHalfEdgeLength_, 0 ); // VertId{3}

    mesh_ = std::make_shared<Mesh>( meshObj );

    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );

    setDirtyFlags( DIRTY_ALL );
}

}
