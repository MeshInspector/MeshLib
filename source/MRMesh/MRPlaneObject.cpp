#include "MRPlaneObject.h"

#include "MRBestFit.h"
#include "MRMatrix3.h"
#include "MRObjectFactory.h"
#include "MRVector3.h"

#include "MRPch/MRJson.h"

namespace
{

using namespace MR;

// return transformation matrix rotating initialBasis to finalBasis
Matrix3f rotateBasis( const Matrix3f& initialBasis, const Matrix3f& finalBasis )
{
    return finalBasis * initialBasis.inverse();
}

} // namespace

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

void PlaneObject::orientateFollowMainAxis_( ViewportId id /*= {}*/ )
{

    auto axis = Vector3f::plusZ();
    auto n = getNormal( id );
    auto planeVectorInXY = cross( axis, n );

    // if plane approx. parallel to XY plane, orentate it using XZ plane
    constexpr float parallelVectorsSinusAngleLimit = 9e-2f; // ~5 degree
    if ( planeVectorInXY.length() < parallelVectorsSinusAngleLimit )
    {
        axis = Vector3f::plusY();
        planeVectorInXY = cross( axis, n );
    }

    planeVectorInXY = planeVectorInXY.normalized();

    auto Y = cross( n, planeVectorInXY );
    Matrix3f bestPlaneBasis;
    bestPlaneBasis.x = Y.normalized();
    bestPlaneBasis.y = planeVectorInXY.normalized();
    bestPlaneBasis.z = n.normalized();

    auto currentPlaneBasis = calcLocalBasis();
    auto A = rotateBasis( currentPlaneBasis, bestPlaneBasis );
    const Matrix3f& r = r_.get( id ), s = s_.get( id );

    auto currXf = xf( id );
    currXf.A = r * A * s;
    setXf( currXf, id );
}

PlaneObject::PlaneObject( const std::vector<Vector3f>& pointsToApprox )
    : PlaneObject()
{
    PointAccumulator pa;
    for ( const auto& p : pointsToApprox )
        pa.addPoint( p );

    auto xf = pa.getBasicXf3f();
    // swap X and Z axes
    xf.A = xf.A * Matrix3f::rotation( Vector3f::plusY(), M_PI_2f );

    Box3f box;
    const auto xfInv = xf.inverse();
    for ( const auto& p : pointsToApprox )
        box.include( xfInv( p ) );

    xf.b = xf( box.center() );
    setXf( xf );

    const auto size = box.size();
    setSizeX( size.x );
    setSizeY( size.y );
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
