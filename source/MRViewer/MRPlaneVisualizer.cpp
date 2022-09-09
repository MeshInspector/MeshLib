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
    objects_ = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( objects_.empty() )
        return;

    Box3f box = Box3f();
    xfChangedConnections_.reserve( objects_.size() );

    for ( const auto& i : objects_ )
    {
        if ( i )
            box.include( i->getWorldBox() );

        xfChangedConnections_.push_back( i->worldXfChangedSignal.connect( [this]
        {
            updatePlane( false );
        } ) );
    }

    setupPlane();
    setupFrameBorder();    
}

PlaneVisualizer::~PlaneVisualizer()
{
    if ( planeObj_ )
        planeObj_->detachFromParent();
    //for ( const auto& obj : objects_ )
      //  obj->setVisualizeProperty( true, VisualizeMaskType::ClippedByPlane, viewer->viewport().id );
    if ( frameBorder_ )
        frameBorder_->detachFromParent();    

    for ( auto& connection : xfChangedConnections_ )
        connection.disconnect();
}

void PlaneVisualizer::setupPlane()
{
    plane_ = Plane3f( Vector3f::plusX(), 0.0f );

    std::shared_ptr<Mesh> planeMesh = std::make_shared<Mesh>( makePlane() );
    planeObj_ = std::make_shared<ObjectMesh>();
    planeObj_->setName( "PlaneObjectClipPlugin" );
    planeObj_->setMesh( planeMesh );
    planeObj_->setAncillary( true );
    planeObj_->setFrontColor( Color( Vector4f::diagonal( 0.3f ) ), false );
    planeObj_->setBackColor( Color( Vector4f::diagonal( 0.3f ) ) );
    planeObj_->setVisible( false );

    updatePlane();

    SceneRoot::get().addChild( planeObj_ );
}

void PlaneVisualizer::updatePlane( bool updateCameraRotation /*= true*/ )
{
    auto viewer = Viewer::instance();

    updateXfs();
    plane_ = plane_.normalized();
    //viewer->viewport().setClippingPlane( plane_ );

    auto trans1 = AffineXf3f::translation( plane_.project( objectsBox_.center() ) );
    auto rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), plane_.n ) );
    auto scale1 = AffineXf3f::linear( Matrix3f::scale( objectsBox_.diagonal() ) );
    AffineXf3f transform = trans1 * rot1 * scale1;
    if ( updateCameraRotation )
        cameraUp3Old_ = viewer->viewport().getUpDirection();
    Vector3f cameraUp3 = cameraUp3Old_;
    auto rot2 = Matrix3f::rotation( transform.A * Vector3f::plusY(),
                                    plane_.project( transform( Vector3f() ) + cameraUp3 ) - transform( Vector3f() ) );

    lastPlaneTransform_ = trans1 * AffineXf3f::linear( rot2 ) * rot1;
    transform = lastPlaneTransform_ * scale1;
    planeObj_->setXf( transform );
}

void PlaneVisualizer::updateXfs()
{
    std::vector<AffineXf3f> objectsWorldXf( objects_.size() );
    objectsBox_ = Box3f();
    for ( int i = 0; i < objects_.size(); ++i )
    {
        auto& obj = objects_[i];
        auto& xf = objectsWorldXf[i];
        xf = obj->worldXf();
        auto box = obj->getBoundingBox();
        objectsBox_.include( xf( box.min ) );
        objectsBox_.include( xf( box.max ) );
    }
}

void PlaneVisualizer::definePlane()
{
    if ( planeIsDefined_ )
        return;

    planeIsDefined_ = true;
    planeObj_->setVisible( true );

    frameBorder_->setVisible( true );

    // for ( const auto& obj : objects_ )
        // obj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, viewer->viewport().id );
}

void PlaneVisualizer::undefinePlane()
{
    if ( !planeIsDefined_ )
        return;
    if ( planeObj_ )
        planeObj_->setVisible( false );
    if ( frameBorder_ )
        frameBorder_->setVisible( false );
    //for ( const auto& obj : objects_ )
//obj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, viewer->viewport().id );
    planeIsDefined_ = false;
}

void PlaneVisualizer::setupFrameBorder()
{
    frameBorder_ = std::make_shared<ObjectLines>();
    frameBorder_->setName( "FrameBorderClipPlugin" );
    frameBorder_->setAncillary( true );
    const Polyline3 polyline( { { Vector2f( 0.5f, 0.5f ), Vector2f( 0.5f, -0.5f ), Vector2f( -0.5f, -0.5f ), Vector2f( -0.5f, 0.5f ), Vector2f( 0.5f, 0.5f ) } } );
    frameBorder_->setPolyline( std::make_shared<Polyline3>( polyline ) );
    frameBorder_->setLinesColorMap( Vector<Color, UndirectedEdgeId>{ 8, Color::yellow() } );
    frameBorder_->setColoringType( ColoringType::LinesColorMap );
    frameBorder_->setVisible( false );

    SceneRoot::get().addChild( frameBorder_ );
}

void PlaneVisualizer::updateFrameBorder()
{
    if ( !objectsBox_.valid() || !frameBorder_ )
        return;

    const AffineXf3f boxScale = AffineXf3f::linear( Matrix3f::scale( objectsBox_.diagonal() ) );
    frameBorder_->setXf( lastPlaneTransform_ * boxScale );
}
}