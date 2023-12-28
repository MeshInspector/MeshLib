#include "MRCylinderObject.h"
#include "MRMatrix3.h"
#include "MRCylinder.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include <Eigen/Dense>
#include "MRCylinderApproximator.h"
#include "MRMeshFwd.h"
#include "MRLine.h"
#include "MRGTest.h"

#include <iostream>
#include "MRMeshNormals.h"
#include "MRMeshSubdivide.h"
namespace
{
constexpr int cDetailLevel = 2048;
constexpr float cBaseRadius = 1.0f;
constexpr float cBaseLength = 1.0f;

constexpr float epsilonForCylinderTopBottomDetection = 0.01f;
constexpr int phiResolution = 180;
constexpr int thetaiResolution = 180;

}

namespace MR
{

MR_ADD_CLASS_FACTORY( CylinderObject )


std::shared_ptr<MR::Mesh> makeFeatureCylinder( int resolution = cDetailLevel, float  startAngle = 0.0f, float  archSize = 2.0f * PI_F )
{
    auto mesh = std::make_shared<MR::Mesh>( makeCylinderAdvanced( cBaseRadius, cBaseRadius, startAngle, archSize, cBaseLength, resolution ) );
    MR::AffineXf3f shift;
    shift.b = MR::Vector3f( 0.0f, 0.0f, -cBaseLength / 2.0f );
    mesh->transform( shift );

    // remove cylinder top and bottom;
    MR::Vector3f zDirection{ 0,0,1 };
    MR::FaceBitSet facesForDelete;
    auto normals = computePerFaceNormals( *mesh );

    for ( auto f : mesh->topology.getValidFaces() )
    {
        if ( MR::cross( normals[f], zDirection ).lengthSq() < epsilonForCylinderTopBottomDetection )
            facesForDelete.autoResizeSet( f, true );
    }
    mesh->topology.deleteFaces( facesForDelete );

    return mesh;
}

float CylinderObject::getLength() const
{
    return xf().A.toScale().z;
}
void CylinderObject::setLength( float length )
{
    auto currentXf = xf();
    auto radius = getRadius();
    currentXf.A = Matrix3f::scale( radius, radius, length );
    setXf( currentXf );
}

float CylinderObject::getRadius() const
{
    return ( xf().A.toScale().x + xf().A.toScale().y ) / 2.0f;
}

void CylinderObject::setRadius( float radius )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::scale( radius, radius, getLength() );
    setXf( currentXf );
}

Vector3f CylinderObject::getDirection() const
{
    return ( xf().A * Vector3f::plusZ() ).normalized();
}

Vector3f CylinderObject::getCenter() const
{
    return xf().b;
}

void CylinderObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
}

void CylinderObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

CylinderObject::CylinderObject()
{
    constructMesh_();
}

CylinderObject::CylinderObject( const std::vector<Vector3f>& pointsToApprox )
{
    // create mesh
    constructMesh_();

    // calculate cylinder parameters.
    MR::Cylinder3<float> result;
    auto fit = Cylinder3Approximation<float>();
    fit.solveGeneral( pointsToApprox, result, phiResolution, thetaiResolution );

    // setup parameters
    setRadius( result.radius );
    setLength( result.length );
    setDirection( result.direction() );
    setCenter( result.center() );
}

std::shared_ptr<Object> CylinderObject::shallowClone() const
{
    auto res = std::make_shared<CylinderObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

std::shared_ptr<Object> CylinderObject::clone() const
{
    auto res = std::make_shared<CylinderObject>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

void CylinderObject::swapBase_( Object& other )
{
    if ( auto cylinderObject = other.asType<CylinderObject>() )
        std::swap( *this, *cylinderObject );
    else
        assert( false );
}

void CylinderObject::serializeFields_( Json::Value& root ) const
{
    ObjectMeshHolder::serializeFields_( root );
    root["Type"].append( CylinderObject::TypeName() );
}

void CylinderObject::constructMesh_()
{
    mesh_ = makeFeatureCylinder();
    setFlatShading( false );
    selectFaces( {} );
    selectEdges( {} );
    setDirtyFlags( DIRTY_ALL );
}




TEST( MRMesh, CylinderApproximation )
{
    float originalRadius = 1.5f;
    float originalLength = 10.0f;
    float startAngle = 0.0f;
    float archSize = PI_F / 1.5f;
    int  resolution = 100;

    MR::AffineXf3f testXf;
    MR::Vector3f center{ 1,2,3 };
    MR::Vector3f direction = ( MR::Vector3f{ 3,2,1 } ).normalized();

    testXf = MR::AffineXf3f::translation( { 1,2,3 } );
    testXf.A = Matrix3f::rotation( Vector3f::plusZ(), direction ) * MR::Matrix3f::scale( { originalRadius , originalRadius  ,originalLength } );

    std::vector<MR::Vector3f> points;

    float angleStep = archSize / resolution;
    float zStep = 1.0f / resolution;
    for ( int i = 0; i < resolution; ++i )
    {
        float angle = startAngle + i * angleStep;
        float z = i * zStep - 0.5f;
        points.emplace_back( testXf( MR::Vector3f{ cosf( angle )  , sinf( angle ) , z } ) );
        points.emplace_back( testXf( MR::Vector3f{ cosf( angle )  , sinf( angle ) ,  -z } ) );
    }

    /////////////////////////////
    // General multithread test 
    /////////////////////////////

    Cylinder3<float> result;
    auto fit = Cylinder3Approximation<float>();
    auto approximationRMS = fit.solveGeneral( points, result, phiResolution, thetaiResolution, true );
    std::cout << "multi thread center: " << result.center() << " direction:" << result.direction() << " length:" << result.length << " radius:" << result.radius << " error:" << approximationRMS << std::endl;

    EXPECT_LE( approximationRMS, 0.1f );
    EXPECT_NEAR( result.radius, originalRadius, 0.1f );
    EXPECT_NEAR( result.length, originalLength, 0.1f );
    EXPECT_LE( ( result.center() - center ).length(), 0.1f );
    EXPECT_GT( MR::dot( direction, result.direction() ), 0.9f );

    ///////////////////////////////////////
    // Compare single thread vs multithread 
    ///////////////////////////////////////

    Cylinder3<float> resultST;
    auto approximationRMS_ST = fit.solveGeneral( points, resultST, phiResolution, thetaiResolution, false );
    std::cout << "single thread center: " << result.center() << " direction:" << result.direction() << " length:" << result.length << " radius:" << result.radius << " error:" << approximationRMS << std::endl;

    EXPECT_NEAR( approximationRMS, approximationRMS_ST, 0.01f );
    EXPECT_NEAR( result.radius, resultST.radius, 0.01f );
    EXPECT_NEAR( result.length, resultST.length, 0.01f );
    EXPECT_LE( ( result.center() - resultST.center() ).length(), 0.01f );
    EXPECT_GT( MR::dot( resultST.direction(), result.direction() ), 0.99f );

    //////////////////////////////////////////
    // Test usage with SpecificAxisFit (SAF)
    //////////////////////////////////////////

    Cylinder3<float> resultSAF;
    MR::Vector3f noice{ 0.002f , -0.003f , 0.01f };

    auto approximationRMS_SAF = fit.solveSpecificAxis( points, resultSAF, direction + noice );
    std::cout << "SpecificAxisFit center: " << resultSAF.center() << " direction:" << resultSAF.direction() << " length:" << resultSAF.length << " radius:" << resultSAF.radius << " error:" << approximationRMS_SAF << std::endl;

    EXPECT_LE( approximationRMS_SAF, 0.1f );
    EXPECT_NEAR( resultSAF.radius, originalRadius, 0.1f );
    EXPECT_NEAR( resultSAF.length, originalLength, 0.1f );
    EXPECT_LE( ( resultSAF.center() - center ).length(), 0.1f );
    EXPECT_GT( MR::dot( direction, resultSAF.direction() ), 0.9f );
}


}