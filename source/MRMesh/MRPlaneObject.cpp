#include "MRPlaneObject.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRBestFit.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3.h"
#include "MRVector3.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( PlaneObject )

Vector3f PlaneObject::getNormal() const
{
    return ( xf().A * Vector3f::plusZ() ).normalized();
}

Vector3f PlaneObject::getCenter() const
{
    return xf().b;
}

void PlaneObject::setNormal( const Vector3f& normal )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
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
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( Vector3f::diagonal( size ) );
    setXf( currentXf );
}

PlaneObject::PlaneObject()
{
    constructMesh_();
}

PlaneObject::PlaneObject( const std::vector<Vector3f>& pointsToApprox )
{
    constructMesh_();

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
    setSize( box.diagonal() * 2.f );
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
    meshObj.points.emplace_back( -1, -1, 0 ); // VertId{0}
    meshObj.points.emplace_back( 1, -1, 0 ); // VertId{1}
    meshObj.points.emplace_back( -1, 1, 0 ); // VertId{2}
    meshObj.points.emplace_back( 1, 1, 0 ); // VertId{3}

    mesh_ = std::make_shared<Mesh>( meshObj );

    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );

    setDirtyFlags( DIRTY_ALL );
}

}
