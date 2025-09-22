#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRMeasurementObject.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include <MRMesh/MRMesh.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRObjectImGuiLabel.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"
#include "MRPch/MRTBB.h"
#include "MRViewportGlobalBasis.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif

namespace MR
{

namespace
{

constexpr Vector3f cameraEye{0.0f,0.0f,5.0f};
constexpr Vector3f cameraUp{0.0f,1.0f,0.0f};
constexpr Vector3f cameraCenter;
const AffineXf3f cXf = lookAt( cameraCenter, cameraEye, cameraUp );

}

AffineXf3f Viewport::getViewXf_() const
{
    AffineXf3f rot = AffineXf3f::linear( Matrix3f( params_.cameraTrackballAngle ) * Matrix3f::scale( params_.cameraZoom ) );
    AffineXf3f tr = AffineXf3f::translation( params_.cameraTranslation );
    return cXf * rot * tr;
}

Vector3f Viewport::getCameraPoint() const
{
    AffineXf3f xfInv = AffineXf3f( viewM_ ).inverse();
    return xfInv.b;
}

void Viewport::setCameraPoint( const Vector3f& cameraWorldPos )
{
    AffineXf3f rot = AffineXf3f::linear( Matrix3f( params_.cameraTrackballAngle ) * Matrix3f::scale( params_.cameraZoom ) );
    params_.cameraTranslation = ( cXf * rot ).inverse().b - cameraWorldPos;
    needRedraw_ = true;
}

void Viewport::setupViewMatrix_()
{
    viewM_ = getViewXf_();

    if ( rotation_ )
        rotateView_();
}

void Viewport::setupProjMatrix_()
{
    if ( params_.orthographic )
    {
        // setup orthographic matrix
        float angle = 0.5f * params_.cameraViewAngle / 180.0f * PI_F;
        float h = tan( angle );
        float d = h * width( viewportRect_ ) / height( viewportRect_ );
        projM_( 0, 0 ) = 1.f / d; projM_( 0, 1 ) = 0.f; projM_( 0, 2 ) = 0.f; projM_( 0, 3 ) = 0.f;
        projM_( 1, 0 ) = 0.f; projM_( 1, 1 ) = 1.f / h; projM_( 1, 2 ) = 0.f; projM_( 1, 3 ) = 0.f;
        projM_( 2, 0 ) = 0.f; projM_( 2, 1 ) = 0.f;
        projM_( 2, 2 ) = -2.f / ( params_.cameraDfar - params_.cameraDnear );
        projM_( 2, 3 ) = -( params_.cameraDfar + params_.cameraDnear ) / ( params_.cameraDfar - params_.cameraDnear );
        projM_( 3, 0 ) = 0.f; projM_( 3, 1 ) = 0.f; projM_( 3, 2 ) = 0.f; projM_( 3, 3 ) = 1.f;
    }
    else
    {
        // setup perspective matrix
        float angle = 0.5f * params_.cameraViewAngle / 180.0f * PI_F;
        float h = tan( angle ) * params_.cameraDnear;
        float d = h * width( viewportRect_ ) / height( viewportRect_ );
        projM_( 0, 0 ) = params_.cameraDnear / d; projM_( 0, 1 ) = 0.f; projM_( 0, 2 ) = 0.f; projM_( 0, 3 ) = 0.f;
        projM_( 1, 0 ) = 0.f; projM_( 1, 1 ) = params_.cameraDnear / h; projM_( 1, 2 ) = 0.f; projM_( 1, 3 ) = 0.f;
        projM_( 2, 0 ) = 0.f; projM_( 2, 1 ) = 0.f;
        projM_( 2, 2 ) = ( params_.cameraDfar + params_.cameraDnear ) / ( params_.cameraDnear - params_.cameraDfar );
        projM_( 2, 3 ) = -2.f * ( params_.cameraDfar * params_.cameraDnear ) / ( params_.cameraDfar - params_.cameraDnear );
        projM_( 3, 0 ) = 0.f; projM_( 3, 1 ) = 0.f; projM_( 3, 2 ) = -1.f; projM_( 3, 3 ) = 0.f;
    }
}

void Viewport::setupAxesProjMatrix_()
{
    float h = 1.0f;// ( cameraEye - cameraCenter ).length();
    float d = h * width( viewportRect_ ) / height( viewportRect_ );
    axesProjMat_( 0, 0 ) = 1.f / d; axesProjMat_( 0, 1 ) = 0.f; axesProjMat_( 0, 2 ) = 0.f; axesProjMat_( 0, 3 ) = 0.f;
    axesProjMat_( 1, 0 ) = 0.f; axesProjMat_( 1, 1 ) = 1.f / h; axesProjMat_( 1, 2 ) = 0.f; axesProjMat_( 1, 3 ) = 0.f;
    axesProjMat_( 2, 0 ) = 0.f; axesProjMat_( 2, 1 ) = 0.f;
    axesProjMat_( 2, 2 ) = -2.f / (params_.cameraDfar - params_.cameraDnear);
    axesProjMat_( 2, 3 ) = -(params_.cameraDfar + params_.cameraDnear) / (params_.cameraDfar - params_.cameraDnear);
    axesProjMat_( 3, 0 ) = 0.f; axesProjMat_( 3, 1 ) = 0.f; axesProjMat_( 3, 2 ) = 0.f; axesProjMat_( 3, 3 ) = 1.f;
}

// ================================================================
// move camera part

void Viewport::setRotation( bool state )
{
    if ( rotation_ == state )
        return;

    needRedraw_ = true;
    rotation_ = state;
    if ( !rotation_ )
        return;

    bool boxUpdated = false;
    if ( !sceneBox_.valid() )
    {
        boxUpdated = true;
        updateSceneBox_();
    }

    bool pickedSuccessfuly_{ false };
    if ( params_.rotationMode != Parameters::RotationCenterMode::Static )
    {
        auto [obj, pick] = pick_render_object();
        pickedSuccessfuly_ = obj && pick.face.valid();
        if ( pickedSuccessfuly_ )
            setRotationPivot_( obj->worldXf()( pick.point ) );
    }
    if ( params_.rotationMode != Parameters::RotationCenterMode::Dynamic && !pickedSuccessfuly_ )
    {
        if ( !boxUpdated )
            updateSceneBox_(); // need update here anyway, but under flag, not to update twice
        auto sceneCenter = sceneBox_.valid() ? sceneBox_.center() : Vector3f();
        setRotationPivot_( params_.staticRotationPivot ? *params_.staticRotationPivot : sceneCenter );
    }

    auto sceneCenter = sceneBox_.valid() ? sceneBox_.center() : Vector3f();
    distToSceneCenter_ = ( getCameraPoint() - sceneCenter ).length();

    Vector3f coord = projectToViewportSpace( rotationPivot_ );
    static_viewport_point = Vector2f( coord[0], coord[1] );
    static_point_ = worldToCameraSpace( rotationPivot_ );
}

void Viewport::rotateView_()
{
    // TODO: try to simplify
    AffineXf3f xf(viewM_);
    Vector3f shift = static_point_ - xf.A * rotationPivot_;
    viewM_.setTranslation( shift );

    if ( params_.compensateRotation )
    {
        auto line = unprojectPixelRay( static_viewport_point );
        line.d = line.d.normalized();

        auto sceneCenter = sceneBox_.valid() ? sceneBox_.center() : Vector3f();

        auto dir = sceneCenter - getCameraPoint();
        auto dirOnMoveProjLength = dot( dir, line.d );

        auto hLengthSq = dir.lengthSq() - sqr( dirOnMoveProjLength );
        auto distHDiffSq = sqr( distToSceneCenter_ ) - hLengthSq;
        // distHDiffSq > 0.0f prevents view matrix degeneration in case of bad rotation center
        auto moveLength = ( distHDiffSq > 0.0f ) ? ( std::sqrt( distHDiffSq ) - dirOnMoveProjLength ) : 0.0f;

        Vector3f transformedDir = xf.A * ( moveLength * line.d );
        shift += transformedDir;
    }
    // changing translation should not really be here, so const cast is OK
    // meanwhile it is because we need to keep distance(camera, scene center) static
    params_.cameraTranslation = Matrix3f( params_.cameraTrackballAngle.inverse() ) * ( shift + cameraEye ) / params_.cameraZoom;
    assert( !std::isnan( params_.cameraTranslation.x ) );
    viewM_.setTranslation( shift );
}

void Viewport::transformView( const AffineXf3f & xf )
{
    auto newAngle = params_.cameraTrackballAngle * Quaternionf( xf.A );
    auto newTrans = xf.A.inverse() * ( params_.cameraTranslation + xf.b );
    if ( params_.cameraTrackballAngle == newAngle &&
         params_.cameraTranslation == newTrans )
        return;
    params_.cameraTrackballAngle = newAngle;
    params_.cameraTranslation = newTrans;

    needRedraw_ = true;
}

float Viewport::getPixelSize() const
{
    return ( tan( params_.cameraViewAngle * MR::PI_F / 360.0f ) * params_.cameraDnear * 2.0f ) / ( height( viewportRect_ ) * params_.cameraZoom );
}

float Viewport::getPixelSizeAtPoint( const Vector3f& worldPoint ) const
{
    Vector4f clipVec = getFullViewportMatrix() * Vector4f( worldPoint.x, worldPoint.y, worldPoint.z, 1 );
    return clipVec.w / projM_.y.y / params_.cameraZoom / ( viewportRect_.max.y - viewportRect_.min.y ) * 2.0f;
}


// ================================================================
// projection part

AffineXf3f Viewport::getUnscaledViewXf() const
{
    // TODO. Try to find better normalize way
    AffineXf3f res( viewM_ );
    res.A.x = res.A.x.normalized();
    res.A.y = res.A.y.normalized();
    res.A.z = res.A.z.normalized();
    return res;
}

Matrix4f Viewport::getFullViewportInversedMatrix() const
{
    // compute inverse in double precision to avoid NaN for very small scales
    return Matrix4f( ( Matrix4d( projM_ ) * Matrix4d( viewM_ ) ).inverse() );
}

Line3f Viewport::unprojectPixelRay( const Vector2f& viewportPoint ) const
{
    auto M = getFullViewportInversedMatrix();
    Vector3f clipNear = viewportSpaceToClipSpace( Vector3f( viewportPoint.x, viewportPoint.y, 0.0f ) );
    auto clipFar = clipNear;
    clipFar.z = 1.0f;
    auto p = M( clipNear );
    auto d = M( clipFar ) - p;
    assert ( !std::isnan( d.x ) );
    return Line3f( p, d );
}

Vector3f Viewport::worldToCameraSpace( const Vector3f& p ) const
{
    return viewM_(p);
}

std::vector<Vector3f> Viewport::worldToCameraSpace( const std::vector<Vector3f>& points ) const
{
    std::vector<Vector3f> res( points.size() );
    auto xf = getViewXf();
    for( int i = 0; i < points.size(); i++ )
    {
        res[i] = xf( points[i] );
    }
    return res;
}

Vector3f Viewport::projectToClipSpace( const Vector3f& worldPoint ) const
{
    return getFullViewportMatrix()(worldPoint);
}

Vector3f Viewport::unprojectFromClipSpace( const Vector3f& clipPoint ) const
{
    return getFullViewportInversedMatrix()(clipPoint);
}

std::vector<Vector3f> Viewport::projectToClipSpace( const std::vector<Vector3f>& worldPoints ) const
{
    std::vector<Vector3f> res(worldPoints.size());
    auto M = getFullViewportMatrix();
    for( int i = 0; i < worldPoints.size(); i++ )
    {
        res[i] = M( worldPoints[i] );
    }
    return res;
}
std::vector<Vector3f> Viewport::unprojectFromClipSpace( const std::vector<Vector3f>& clipPoints ) const
{
    std::vector<Vector3f> res( clipPoints.size() );
    auto M = getFullViewportInversedMatrix();
    for( int i = 0; i < clipPoints.size(); i++ )
    {
        res[i] = M( clipPoints[i] );
    }
    return res;
}

Vector3f Viewport::projectToViewportSpace( const Vector3f& worldPoint ) const
{
    auto res = projectToClipSpace( worldPoint );
    return clipSpaceToViewportSpace( res );
}
Vector3f Viewport::unprojectFromViewportSpace( const Vector3f& viewportPoint ) const
{
    auto clipPoint = viewportSpaceToClipSpace( viewportPoint );
    return unprojectFromClipSpace( clipPoint );
}

std::vector<Vector3f> Viewport::projectToViewportSpace( const std::vector<Vector3f>& worldPoints ) const
{
    std::vector<Vector3f> res( worldPoints.size() );
    auto M = getFullViewportMatrix();
    for( int i = 0; i < worldPoints.size(); i++ )
    {
        res[i] = clipSpaceToViewportSpace( M( worldPoints[i] ) );
    }
    return res;
}
std::vector<Vector3f> Viewport::unprojectFromViewportSpace( const std::vector<Vector3f>& worldPoints ) const
{
    std::vector<Vector3f> res(worldPoints.size());
    auto M = getFullViewportInversedMatrix();
    for( int i = 0; i < worldPoints.size(); i++ )
    {
        auto clipPoint = viewportSpaceToClipSpace( worldPoints[i] );
        res[i] = M( clipPoint );
    }
    return res;
}

Vector3f Viewport::clipSpaceToViewportSpace( const Vector3f& p ) const
{
    auto x = ( p.x / 2.f + 0.5f ) * width( viewportRect_ );
    auto y = ( -p.y / 2.f + 0.5f ) * height( viewportRect_ );
    auto z = p.z / 2.f + 0.5f;
    return Vector3f( x, y, z );
}

std::vector<Vector3f> Viewport::clipSpaceToViewportSpace( const std::vector<Vector3f>& points ) const
{
    std::vector<Vector3f> res( points.size() );
    for( int i = 0; i < points.size(); i++ )
    {
        const auto& p = points[i];
        res[i] = clipSpaceToViewportSpace( p );
    }
    return res;
}

Vector3f Viewport::viewportSpaceToClipSpace( const Vector3f& p ) const
{
    auto x = 2.f * p.x / width( viewportRect_ ) - 1.f;
    auto y = -2.f * p.y / height( viewportRect_ ) + 1.f;
    auto z = 2.f * p.z - 1.f;
    return Vector3f( x, y, z );
}

std::vector<Vector3f> Viewport::viewportSpaceToClipSpace( const std::vector<Vector3f>& points ) const
{
    std::vector<Vector3f> res( points.size() );
    for( int i = 0; i < points.size(); i++ )
    {
        const auto& p = points[i];
        res[i] = viewportSpaceToClipSpace( p );
    }
    return res;
}

// ================================================================
// fit data part

void Viewport::preciseFitBoxToScreenBorder( const FitBoxParams& fitParams )
{
    preciseFitToScreenBorder_( [&] ( bool zoomFov, bool globalBasis )->Box3f
    {
        if ( globalBasis )
            return {}; // do not take global basis into account for box fitting (only fit given box)
        Space space = Space::CameraOrthographic;
        if ( !params_.orthographic )
        {
            space = zoomFov ? Space::CameraPerspective : Space::World;
        }
        if ( space == Space::World )
            return fitParams.worldBox;
        else if ( space == Space::CameraOrthographic )
            return transformed( fitParams.worldBox, getViewXf_() );
        else
        {
            assert( space == Space::CameraPerspective );
            auto xf = getViewXf_();
            Box3f res;
            for ( const auto& p : getCorners( fitParams.worldBox ) )
            {
                auto v = xf( p );
                if ( v.z == 0 )
                    continue;
                res.include( Vector3f( v.x / v.z, v.y / v.z, v.z ) );
            }
            return res;
        }
    }, fitParams );
}

void Viewport::preciseFitDataToScreenBorder( const FitDataParams& fitParams )
{
    std::vector<std::shared_ptr<VisualObject>> allObj;
    if ( fitParams.mode == FitMode::CustomObjectsList )
    {
        allObj = fitParams.objsList;
    }
    else
    {
        ObjectSelectivityType type = ObjectSelectivityType::Any;
        if ( fitParams.mode == FitMode::SelectedObjects )
        {
            type = ObjectSelectivityType::Selected;
        }
        else if ( fitParams.mode == FitMode::SelectableObjects )
        {
            type = ObjectSelectivityType::Selectable;
        }

        allObj = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), type );
    }

    preciseFitToScreenBorder_( [&] ( bool zoomFov, bool globalBasis )
    {
        Space space = Space::CameraOrthographic;
        if ( !params_.orthographic )
        {
            space = zoomFov ? Space::CameraPerspective : Space::World;
        }
        if ( !globalBasis )
            return calcBox_( allObj, space, fitParams.mode == FitMode::SelectedPrimitives );
        else
            return calcBox_( getViewerInstance().globalBasis->axesChildren(), space, fitParams.mode == FitMode::SelectedPrimitives );
    }, fitParams );
}

void Viewport::preciseFitToScreenBorder_( std::function<Box3f( bool zoomFOV, bool globalBasis )> getBoxFn, const BaseFitParams& fitParams )
{
    if ( fitParams.snapView )
        params_.cameraTrackballAngle = getClosestCanonicalQuaternion( params_.cameraTrackballAngle );

    const auto safeZoom = params_.cameraZoom;
    params_.cameraZoom = 1;

    Box3f sceneObjsBox = getBoxFn( false, false );
    Box3f unitedBox;
    if ( getViewerInstance().globalBasis && getViewerInstance().globalBasis->isVisible( id ) )
        unitedBox = getBoxFn( false, true ); // calculate box of global basis separately, not to interfere with actual scene size
    unitedBox.include( sceneObjsBox );

    if ( !unitedBox.valid() )
    {
        params_.cameraZoom = safeZoom;
        setRotationPivot_( params_.staticRotationPivot ? *params_.staticRotationPivot : Vector3f() );
        return;
    }

    if ( params_.orthographic )
    {
        sceneBox_ = transformed( unitedBox, getViewXf_().inverse() );
    }
    else
    {
        sceneBox_ = unitedBox;
    }
    Vector3f sceneCenter = params_.orthographic ?
        getViewXf_().inverse()( unitedBox.center() ) : unitedBox.center();
    setRotationPivot_( params_.staticRotationPivot ? *params_.staticRotationPivot : sceneCenter );

    params_.cameraTranslation = -sceneCenter;
    params_.cameraViewAngle = 45.0f;
    params_.objectScale = sceneObjsBox.valid() ? sceneObjsBox.diagonal() : 1.0f; // we should not take global basis into account here
    if ( params_.objectScale == 0.0f )
        params_.objectScale = 1.0f;

    auto unitedSceneScale = unitedBox.diagonal();

    if ( params_.orthographic )
    {
        auto factor = 1.f / ( cameraEye - cameraCenter ).length();
        auto tanFOV = tan( 0.5f * params_.cameraViewAngle / 180.f * PI_F );
        params_.cameraZoom = factor / ( unitedSceneScale * tanFOV );

        const auto winRatio = getRatio();
        auto dX = ( unitedBox.max.x - unitedBox.min.x ) / 2.f / winRatio;
        auto dY = ( unitedBox.max.y - unitedBox.min.y ) / 2.f;
        float maxD = std::max( dX, dY );

        if ( maxD == 0.0f )
            maxD = 1.0f;

        params_.cameraViewAngle = ( 2.f * atan2( maxD * params_.cameraZoom, params_.cameraDnear ) ) / PI_F * 180.f / fitParams.factor;
    }
    else
    {
        auto tanFOV = tan( 0.5f * params_.cameraViewAngle / 180.f * PI_F );
        params_.cameraZoom = 1 / ( unitedSceneScale * tanFOV );

        auto res = getZoomFOVtoScreen_( [&] ()
        {
            auto localSceneObjBox = getBoxFn( true, false );
            Box3f localUnitedBox;
            if ( getViewerInstance().globalBasis && getViewerInstance().globalBasis->isVisible( id ) )
                localUnitedBox = getBoxFn( true, true );
            localUnitedBox.include( localSceneObjBox );
            return localUnitedBox;
        } );
        if ( res.first == 0.0f )
            res.first = 1.0f;
        params_.cameraViewAngle = res.first / fitParams.factor;
    }

    needRedraw_ = true;
}

using ObjectToCameraFunc = std::function<bool ( Vector3f& )>;

class LimitCalc
{
public:
    LimitCalc( const VertCoords & points, const VertBitSet & vregion, const ObjectToCameraFunc& obj2cam )
        : points_( points ), vregion_( vregion ), obj2cam_( obj2cam ) { }
    LimitCalc( LimitCalc& x, tbb::split )
        : points_( x.points_ ), vregion_( x.vregion_ ), obj2cam_( x.obj2cam_ ) { }
    void join( const LimitCalc& y ) { box_.include( y.box_ ); }

    const Box3f& box() const { return box_; }

    void operator ()( const tbb::blocked_range<VertId>& r )
    {
        for ( auto v = r.begin(); v < r.end(); ++v )
        {
            if ( !vregion_.test( v ) )
                continue;
            auto pt = points_[v];
            if ( obj2cam_( pt ) )
                box_.include( pt );
        }
    }

private:
    const VertCoords & points_;
    const VertBitSet & vregion_;
    const ObjectToCameraFunc& obj2cam_;
    Box3f box_;
};

bool Viewport::allModelsInsideViewportRectangle() const
{
    auto res = getZoomFOVtoScreen_( [&] ()
    {
        return calcBox_( getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any ), params_.orthographic ? Space::CameraOrthographic : Space::CameraPerspective );
    } );
    return res.second && params_.cameraViewAngle > res.first;
}

Box3f Viewport::calcBox_( const std::vector<std::shared_ptr<VisualObject>>& objs, Space space, bool selectedPrimitives /*= false*/ ) const
{
    Box3f box;

    // object space to camera space
    const auto makeObj2Cam = [space, xfV = getViewXf_()] ( const AffineXf3f& xfObj ) -> ObjectToCameraFunc
    {
        switch ( space )
        {
        case Space::World:
            return [xf = xfObj] ( Vector3f& p )
            {
                p = xf( p );
                return true;
            };
        case Space::CameraOrthographic:
            return [xf = xfV * xfObj] ( Vector3f& p )
            {
                p = xf( p );
                return true;
            };
        case Space::CameraPerspective:
            return [xf = xfV * xfObj] ( Vector3f& p )
            {
                auto v = xf( p );
                if ( v.z == 0 )
                    return false;
                p = Vector3f( v.x / v.z, v.y / v.z, v.z );
                return true;
            };
        }
        MR_UNREACHABLE
    };

    const auto expandBox = [&box] ( const VertCoords& coords, const VertBitSet& region, const ObjectToCameraFunc& obj2cam )
    {
        LimitCalc calc( coords, region, obj2cam );
        parallel_reduce( tbb::blocked_range( region.find_first(), region.find_last() + 1 ), calc );
        box.include( calc.box() );
    };

    for ( const auto& obj : objs )
    {
        if ( !obj->globalVisibility( id ) )
            continue;

        const auto xf = obj->worldXf( id );
        const auto obj2cam = makeObj2Cam( xf );

        if ( selectedPrimitives )
        {
            if ( auto* objMesh = obj->asType<ObjectMeshHolder>() )
            {
                if ( !objMesh->mesh() )
                    continue;

                const auto& mesh = *objMesh->mesh();
                const auto region =
                    getIncidentVerts( mesh.topology, objMesh->getSelectedEdges() )
                    | getIncidentVerts( mesh.topology, objMesh->getSelectedFaces() );
                if ( region.any() )
                    expandBox( mesh.points, region, obj2cam );
            }

            continue;
        }

        if ( auto* objMesh = obj->asType<ObjectMeshHolder>() )
        {
            if ( !objMesh->mesh() )
                continue;

            const auto& mesh = *objMesh->mesh();
            expandBox( mesh.points, mesh.topology.getValidVerts(), obj2cam );
        }
        else if ( auto* objLines = obj->asType<ObjectLinesHolder>() )
        {
            if ( !objLines->polyline() )
                continue;

            const auto& polyline = *objLines->polyline();
            expandBox( polyline.points, polyline.topology.getValidVerts(), obj2cam );
        }
        else if ( auto objPoints = obj->asType<ObjectPointsHolder>() )
        {
            if ( !objPoints->pointCloud() )
                continue;

            const auto& pointCloud = *objPoints->pointCloud();
            expandBox( pointCloud.points, pointCloud.validPoints, obj2cam );
        }
#ifndef MRVIEWER_NO_VOXELS
        else if ( auto* objVox = obj->asType<ObjectVoxels>(); objVox && objVox->isVolumeRenderingEnabled() )
        {
            if ( !objVox->grid() )
                continue;

            const auto& vdbVolume = objVox->vdbVolume();
            Box3f voxBox;
            voxBox.include( Vector3f() );
            voxBox.include( mult( Vector3f( vdbVolume.dims ), vdbVolume.voxelSize ) );

            for ( auto p : getCorners( voxBox ) )
                if ( obj2cam( p ) )
                    box.include( p );
        }
#endif
        else if ( const auto objBox = obj->getBoundingBox(); objBox.valid() )
        {
            for ( auto p : getCorners( objBox ) )
                if ( obj2cam( p ) )
                    box.include( p );
        }
    }
    return box;
}

std::pair<float, bool> Viewport::getZoomFOVtoScreen_( std::function<Box3f()> getBoxFn, Vector3f* cameraShift /*= nullptr*/ ) const
{
    //const auto box = calcBox_( objs, params_.orthographic ? Space::CameraOrthographic : Space::CameraPerspective, selectedPrimitives );
    const auto box = getBoxFn();
    if ( !box.valid() )
        return std::make_pair( params_.cameraViewAngle, true );

    // z points from scene to camera
    bool allInside = (-box.max.z < params_.cameraDfar) && (-box.min.z > params_.cameraDnear);

    const auto winRatio = getRatio();
    if( params_.orthographic && cameraShift )
    {
        auto dX = ( box.max.x - box.min.x ) / 2.f / winRatio;
        auto dY = ( box.max.y - box.min.y ) / 2.f;
        float maxD = std::max( dX, dY );
        auto meanX = ( box.max.x + box.min.x ) / 2.f;
        auto meanY = ( box.max.y + box.min.y ) / 2.f;
        const AffineXf3f xfV = getViewXf_();
        *cameraShift = -xfV.A.x.normalized() * ( meanX / params_.cameraZoom ) - xfV.A.y.normalized() * ( meanY / params_.cameraZoom );
        return std::make_pair( ( 2.f * atan2( maxD, params_.cameraDnear ) ) / PI_F * 180.f, allInside );
    }
    else if ( params_.orthographic )
    {
        auto maxX = std::max( box.max.x, -box.min.x ) / winRatio;
        auto maxY = std::max( box.max.y, -box.min.y );
        return std::make_pair( ( 2.f * atan2( std::max( maxX, maxY ), params_.cameraDnear ) ) / PI_F * 180.f, allInside );
    }
    else
    {
        assert( cameraShift == nullptr );
        auto maxX = std::max( box.max.x, -box.min.x ) / winRatio;
        auto maxY = std::max( box.max.y, -box.min.y );
        return std::make_pair( (2.f * atan( std::max( maxX, maxY ) ) ) / PI_F * 180.f, allInside );
    }
}

void Viewport::fitBox( const Box3f& newSceneBox, float fill /*= 1.0f*/, bool snapView /*= true */ )
{
    sceneBox_ = newSceneBox;
    if ( !newSceneBox.valid() )
    {
        setRotationPivot_( params_.staticRotationPivot ? *params_.staticRotationPivot : Vector3f() );
        return;
    }
    auto sceneCenter = sceneBox_.center();
    setRotationPivot_( params_.staticRotationPivot ? *params_.staticRotationPivot : sceneCenter );
    params_.cameraTranslation = -sceneCenter;

    auto dif = sceneBox_.max - sceneBox_.min;
    params_.cameraViewAngle = 45.0f;
    params_.objectScale = dif.length();
    if ( params_.objectScale == 0.0f )
        params_.objectScale = 1.0f;

    auto tanFOV = tan( 0.5f * params_.cameraViewAngle / 180.f * PI_F );
    auto factor = params_.orthographic ? 1.f / ( cameraEye - cameraCenter ).length() : 1.f;
    params_.cameraZoom = factor * fill / ( params_.objectScale * tanFOV );

    if ( snapView )
        params_.cameraTrackballAngle = getClosestCanonicalQuaternion( params_.cameraTrackballAngle );

    needRedraw_ = true;
}

void Viewport::fitData( float fill, bool snapView )
{
    updateSceneBox_();
    fitBox( sceneBox_, fill, snapView );
}

// ================================================================
// set-get parameters part

float Viewport::getRatio() const
{
    // tan([left-right] / 2) / tan([up-down] / 2)
    return width( viewportRect_ ) / height( viewportRect_ );
}

void Viewport::setCameraTrackballAngle( const Quaternionf& rot )
{
    if ( params_.cameraTrackballAngle == rot )
        return;
    params_.cameraTrackballAngle = rot;
    needRedraw_ = true;
}

void Viewport::setCameraTranslation( const Vector3f& translation )
{
    if ( params_.cameraTranslation == translation )
        return;
    params_.cameraTranslation = translation;

    needRedraw_ = true;
}

void Viewport::setCameraViewAngle( float newViewAngle )
{
    if ( params_.cameraViewAngle == newViewAngle )
        return;
    params_.cameraViewAngle = newViewAngle;
    needRedraw_ = true;
}

void Viewport::setCameraZoom( float zoom )
{
    if ( params_.cameraZoom == zoom )
        return;
    params_.cameraZoom = zoom;
    needRedraw_ = true;
}

void Viewport::setOrthographic( bool orthographic )
{
    if ( params_.orthographic == orthographic )
        return;

    params_.orthographic = orthographic;
    preciseFitDataToScreenBorder( { 0.9f } );
    needRedraw_ = true;
}

void Viewport::cameraLookAlong( const Vector3f& newDir, const Vector3f& up )
{
    assert( std::abs( dot( newDir.normalized(), up.normalized() ) ) < 1e-6f );

    Vector3f lookDir = cameraCenter - cameraEye;
    auto rotLook = Matrix3f::rotation( newDir, lookDir );

    auto upVec = rotLook.inverse() * cameraUp;
    float sign = 1.0f;
    if ( dot( cross( up, upVec ), newDir ) < 0.0f )
        sign = -1.0f;

    auto rotUp = Matrix3f::rotation( newDir, sign * angle( up, upVec ) );

    params_.cameraTrackballAngle = rotLook;
    params_.cameraTrackballAngle = params_.cameraTrackballAngle * Quaternionf( rotUp );

    needRedraw_ = true;
}

void Viewport::cameraRotateAround( const Line3f& axis, float angle )
{
    // store pivot in camera space
    Vector3f pivot = worldToCameraSpace( axis.p );

    // find rotation around camera center
    params_.cameraTrackballAngle = params_.cameraTrackballAngle * Quaternionf( axis.d.normalized(), -angle );

    // find shift in camera space
    AffineXf3f worldToCameraXf = getViewXf_();
    Vector3f shift = pivot - worldToCameraXf( axis.p );

    // convert and set shift from camera space to world
    params_.cameraTranslation += worldToCameraXf.inverse().A * ( shift );

    needRedraw_ = true;
}

void Viewport::draw_rotation_center() const
{
    if ( !rotation_ || !Viewer::constInstance()->rotationSphere->isVisible( id ) )
        return;

    auto factor = params_.orthographic ? 0.1f / (cameraEye - cameraCenter).length() : 0.1f;
    Viewer::constInstance()->rotationSphere->setXf( AffineXf3f::translation( rotationPivot_ ) *
        AffineXf3f::linear( Matrix3f::scale(factor * tan( params_.cameraViewAngle / 360.0f * PI_F ) / params_.cameraZoom ) ) );

    draw( *Viewer::constInstance()->rotationSphere, Viewer::constInstance()->rotationSphere->worldXf() );
}

}
