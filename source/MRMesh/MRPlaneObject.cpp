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

// Offset in positive and negative directions along the X and Y axes when constructing a base object.
// Historically it eq. 1,  which means that original plane have a 2x2 size.
// basePlaneObjectHalfEdgeLength_=0.5 looks better.
// But left as is for compatibility.
constexpr float basePlaneObjectHalfEdgeLength_ = 1.0f;

MR_ADD_CLASS_FACTORY( PlaneObject )

Vector3f PlaneObject::getNormal( ViewportId id /*= {}*/ ) const
{
    return ( r_.get( id ) * Vector3f::plusZ() ).normalized();
}

Vector3f PlaneObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void PlaneObject::setNormal( const Vector3f& normal, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * s_.get( id );
    setXf( currentXf, id );
    orientateFollowMainAxis_( id );

}

void PlaneObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}

void PlaneObject::setSize( float size, ViewportId id /*= {}*/ )
{
    auto xSize = getSizeX( id );
    auto ySize = getSizeY( id );

    setSizeX( 2.0f * size / ( 1.0f + ySize / xSize ), id );
    setSizeY( 2.0f * size / ( 1.0f + xSize / ySize ), id );
}

void PlaneObject::setSizeX( float size, ViewportId id /*= {}*/ )
{
    size = size / basePlaneObjectHalfEdgeLength_ / 2.0f; // normalization for base figure dimentions
    auto currentXf = xf( id );
    const auto& s = s_.get( id );
    currentXf.A = r_.get( id ) * Matrix3f::scale( size, s.y.y, ( s.y.y + size ) / 2.0f ); // z-scale need for correct plane normal display.
    setXf( currentXf, id );
}

void PlaneObject::setSizeY( float size, ViewportId id /*= {}*/ )
{
    size = size / basePlaneObjectHalfEdgeLength_ / 2.0f; // normalization for base figure dimentions
    auto currentXf = xf( id );
    Matrix3f r = r_.get( id ), s = s_.get( id );
    currentXf.A = r * Matrix3f::scale( s.x.x, size, ( s.x.x + size ) / 2.0f ); // z-scale need for correct plane normal display.
    setXf( currentXf, id );
}

Vector3f PlaneObject::getBasePoint( ViewportId id /*= {} */ ) const
{
    auto basis = calcLocalBasis( id );
    return getCenter( id ) - basis.x * getSizeX( id ) * 0.5f - basis.y * getSizeY( id ) * 0.5f;
}

FeatureObjectProjectPointResult PlaneObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    const Vector3f& center = getCenter( id );
    const Vector3f& normal = getNormal( id );

    Plane3f plane( normal, dot( normal, center ) );
    auto projection = plane.project( point );

    return { projection, normal };
}


float PlaneObject::getSize( ViewportId id /*= {}*/ ) const
{
    return  ( getSizeX( id ) + getSizeY( id ) ) / 2.0f;
}

float PlaneObject::getSizeX( ViewportId id /*= {}*/ ) const
{
    return  s_.get( id ).x.x * basePlaneObjectHalfEdgeLength_ * 2.0f;
}

float PlaneObject::getSizeY( ViewportId id /*= {}*/ ) const
{
    return  s_.get( id ).y.y * basePlaneObjectHalfEdgeLength_ * 2.0f;
}

Matrix3f PlaneObject::calcLocalBasis( ViewportId id /*= {}*/ ) const
{
    Matrix3f result;
    result.x = ( r_.get( id ) * Vector3f::plusX() ).normalized();
    result.y = ( r_.get( id ) * Vector3f::plusY() ).normalized();
    result.z = ( r_.get( id ) * Vector3f::plusZ() ).normalized();
    return result;
}

const std::vector<FeatureObjectSharedProperty>& PlaneObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
        {"Center", FeaturePropertyKind::position,        &PlaneObject::getCenter, &PlaneObject::setCenter},
        {"Normal", FeaturePropertyKind::direction,       &PlaneObject::getNormal, &PlaneObject::setNormal},
        {"Size"  , FeaturePropertyKind::linearDimension, &PlaneObject::getSize,   &PlaneObject::setSize  },
        {"SizeX" , FeaturePropertyKind::linearDimension, &PlaneObject::getSizeX,  &PlaneObject::setSizeX  },
        {"SizeY" , FeaturePropertyKind::linearDimension, &PlaneObject::getSizeY,  &PlaneObject::setSizeY  },
    };
    return ret;
}

PlaneObject::PlaneObject()
    : FeatureObject( 2 )
{}

// returns transformation matris which rotate initialBasis to finalBasis
Matrix3f rotateBasis( Matrix3f initialBasis, Matrix3f finalBasis )
{
    return finalBasis * initialBasis.inverse();
}

void PlaneObject::orientateFollowMainAxis_( ViewportId id /*= {}*/ )
{
    const auto n = getNormal( id ).normalized();
    auto planeVectorInXY = cross( Vector3f::plusZ(), n );

    // if plane approx. parallel to XY plane, orientate it using XZ plane
    constexpr float parallelVectorsSinusAngleLimitSq = sqr( 9e-2f ); // ~5 degree
    if ( planeVectorInXY.lengthSq() < parallelVectorsSinusAngleLimitSq )
    {
        planeVectorInXY = cross( Vector3f::plusY(), n );
        assert( planeVectorInXY.lengthSq() >= parallelVectorsSinusAngleLimitSq );
    }

    Matrix3f bestPlaneBasis;
    bestPlaneBasis.x = cross( planeVectorInXY, n ).normalized();
    bestPlaneBasis.y = planeVectorInXY.normalized();
    bestPlaneBasis.z = n;
    assert( bestPlaneBasis.det() > 0 );

    auto currentPlaneBasis = calcLocalBasis();
    assert( currentPlaneBasis.det() > 0 );

    auto A = rotateBasis( currentPlaneBasis, bestPlaneBasis );
    assert( A.det() > 0 );

    const Matrix3f& r = r_.get( id ), &s = s_.get( id );
    assert( r.det() > 0 );
    assert( s.det() > 0 );

    auto currXf = xf( id );
    currXf.A = r * A * s;
    assert( currXf.A.det() > 0 );
    setXf( currXf, id );
}

void PlaneObject::setupPlaneSize2DByOriginalPoints_( const std::vector<Vector3f>& pointsToApprox )
{
    Matrix3f r = r_.get();

    MR::Vector3f min( std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 1.0f );
    MR::Vector3f max( -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -1.0f );

    // calculate feature oX and oY direction in world (parent) coordinate system.
    auto oX = ( r * Vector3f::plusX() ).normalized();
    auto oY = ( r * Vector3f::plusY() ).normalized();

    // calculate 2D bounding box in oX, oY coordinate.
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
    }

    // setup sizes
    auto sX = std::abs( max.x - min.x );
    auto sY = std::abs( max.y - min.y );

    setSizeX( sX );
    setSizeY( sY );
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

    // make a normal planeVectorInXY from center directed against a point (0, 0, 0)
    Plane3f plane = pa.getBestPlanef();
    Vector3f normal = plane.n.normalized();
    if ( plane.d < 0 )
        normal *= -1.f;

    setNormal( normal );

    setCenter( plane.project( box.center() ) );
    setupPlaneSize2DByOriginalPoints_( pointsToApprox );
}

std::shared_ptr<Object> PlaneObject::shallowClone() const
{
    return std::make_shared<PlaneObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> PlaneObject::clone() const
{
    return std::make_shared<PlaneObject>( ProtectedStruct{}, *this );
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
    FeatureObject::serializeFields_( root );
    root["Type"].append( PlaneObject::TypeName() );
}

void PlaneObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

}
