#include "MRPlaneObject.h"
#include "MRMesh.h"
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
    return ( xf().A * Vector3f::plusZ() ).normalized();
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

float PlaneObject::getSize( void ) const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return  s.x.x * basePlaneObjectHalfEdgeLength_ * 2.0f;
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
    constructMesh_();
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
    setSize( box.diagonal() );
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
    meshObj.points.emplace_back(  basePlaneObjectHalfEdgeLength_, -basePlaneObjectHalfEdgeLength_, 0 ); // VertId{1}
    meshObj.points.emplace_back( -basePlaneObjectHalfEdgeLength_,  basePlaneObjectHalfEdgeLength_, 0 ); // VertId{2}
    meshObj.points.emplace_back(  basePlaneObjectHalfEdgeLength_,  basePlaneObjectHalfEdgeLength_, 0 ); // VertId{3}

    mesh_ = std::make_shared<Mesh>( meshObj );

    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );

    setDirtyFlags( DIRTY_ALL );
}

}
