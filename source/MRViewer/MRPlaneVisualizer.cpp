#include "MRPlaneVisualizer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRMakePlane.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"

namespace MR
{

PlaneVisualizer::PlaneVisualizer()
{   
    objects = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( objects.empty() )
        return;

    Box3f box = Box3f();
    xfChangedConnections.reserve( objects.size() );

    for ( const auto& i : objects )
    {
        if ( i )
            box.include( i->getWorldBox() );

        xfChangedConnections.push_back( i->worldXfChangedSignal.connect( [this]
        {
            updatePlane( false );
        } ) );
    }

    setupPlane();
    setupFrameBorder();    
}

PlaneVisualizer::~PlaneVisualizer()
{
    if ( planeObj )
        planeObj->detachFromParent();

    for ( const auto& obj : objects )
        obj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, Viewer::instance()->viewport().id );

    if ( frameBorder )
        frameBorder->detachFromParent();    

    for ( auto& connection : xfChangedConnections )
        connection.disconnect();
}

void PlaneVisualizer::setupPlane()
{
    plane = Plane3f( Vector3f::plusX(), 0.0f );

    std::shared_ptr<Mesh> planeMesh = std::make_shared<Mesh>( makePlane() );
    planeObj = std::make_shared<ObjectMesh>();
    planeObj->setName( "PlaneObjectClipPlugin" );
    planeObj->setMesh( planeMesh );
    planeObj->setAncillary( true );
    planeObj->setFrontColor( Color( Vector4f::diagonal( 0.3f ) ), false );
    planeObj->setBackColor( Color( Vector4f::diagonal( 0.3f ) ) );
    planeObj->setVisible( false );

    updatePlane();

    SceneRoot::get().addChild( planeObj );
}

void PlaneVisualizer::updatePlane( bool updateCameraRotation /*= true*/ )
{
    auto viewer = Viewer::instance();

    updateXfs();
    plane = plane.normalized();
    if ( clipByPlane )
        viewer->viewport().setClippingPlane( plane );

    auto trans1 = AffineXf3f::translation( plane.project( objectsBox.center() ) );
    auto rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), plane.n ) );
    auto scale1 = AffineXf3f::linear( Matrix3f::scale( objectsBox.diagonal() ) );
    AffineXf3f transform = trans1 * rot1 * scale1;
    if ( updateCameraRotation )
        cameraUp3Old = viewer->viewport().getUpDirection();
    Vector3f cameraUp3 = cameraUp3Old;
    auto rot2 = Matrix3f::rotation( transform.A * Vector3f::plusY(),
                                    plane.project( transform( Vector3f() ) + cameraUp3 ) - transform( Vector3f() ) );

    lastPlaneTransform = trans1 * AffineXf3f::linear( rot2 ) * rot1;
    transform = lastPlaneTransform * scale1;
    planeObj->setXf( transform );
}

void PlaneVisualizer::updateXfs()
{
    std::vector<AffineXf3f> objectsWorldXf( objects.size() );
    objectsBox = Box3f();
    for ( int i = 0; i < objects.size(); ++i )
    {
        auto& obj = objects[i];
        auto& xf = objectsWorldXf[i];
        xf = obj->worldXf();
        auto box = obj->getBoundingBox();
        objectsBox.include( xf( box.min ) );
        objectsBox.include( xf( box.max ) );
    }
}

void PlaneVisualizer::definePlane()
{
    if ( planeIsDefined )
        return;

    planeIsDefined = true;
    planeObj->setVisible( true );

    frameBorder->setVisible( true );

    if ( clipByPlane )
        for ( const auto& obj : objects )
            obj->setVisualizeProperty( true, VisualizeMaskType::ClippedByPlane, Viewer::instance()->viewport().id );
}

void PlaneVisualizer::undefinePlane()
{
    if ( !planeIsDefined )
        return;
    if ( planeObj )
        planeObj->setVisible( false );
    if ( frameBorder )
        frameBorder->setVisible( false );

    for ( const auto& obj : objects )
        obj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, Viewer::instance()->viewport().id );

    planeIsDefined = false;
}

void PlaneVisualizer::setupFrameBorder()
{
    frameBorder = std::make_shared<ObjectLines>();
    frameBorder->setName( "FrameBorderClipPlugin" );
    frameBorder->setAncillary( true );
    const Polyline3 polyline( { { Vector2f( 0.5f, 0.5f ), Vector2f( 0.5f, -0.5f ), Vector2f( -0.5f, -0.5f ), Vector2f( -0.5f, 0.5f ), Vector2f( 0.5f, 0.5f ) } } );
    frameBorder->setPolyline( std::make_shared<Polyline3>( polyline ) );
    frameBorder->setLinesColorMap( Vector<Color, UndirectedEdgeId>{ 8, Color::yellow() } );
    frameBorder->setColoringType( ColoringType::LinesColorMap );
    frameBorder->setVisible( false );

    SceneRoot::get().addChild( frameBorder );
}

void PlaneVisualizer::updateFrameBorder()
{
    if ( !objectsBox.valid() || !frameBorder )
        return;

    const AffineXf3f boxScale = AffineXf3f::linear( Matrix3f::scale( objectsBox.diagonal() ) );
    frameBorder->setXf( lastPlaneTransform * boxScale );
}
}