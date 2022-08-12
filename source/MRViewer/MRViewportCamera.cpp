#include "MRViewport.h"
#include "MRViewer.h"
#include <MRMesh/MRMesh.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRPch/MRSpdlog.h"
#include <tbb/parallel_reduce.h>

namespace
{

constexpr float cMaxObjectScale = 1e9;

}

namespace MR
{

AffineXf3f Viewport::getViewXf_() const
{
    AffineXf3f c = lookAt( params_.cameraCenter, params_.cameraEye, params_.cameraUp );
    Matrix3f rot = Matrix3f( params_.cameraTrackballAngle ) * Matrix3f::scale( params_.cameraZoom );
    AffineXf3f factor( rot, rot * params_.cameraTranslation );
    return c * factor;
}

void Viewport::setupViewMatrix() const
{
    viewM = getViewXf_();

    if ( rotation_ )
        rotateView_();
}

void Viewport::setupProjMatrix() const
{
    if ( params_.orthographic )
    {
        // setup orthographic matrix
        float angle = 0.5f * params_.cameraViewAngle / 180.0f * PI_F;
        float h = tan( angle );
        float d = h * width( viewportRect_ ) / height( viewportRect_ );
        projM( 0, 0 ) = 1.f / d; projM( 0, 1 ) = 0.f; projM( 0, 2 ) = 0.f; projM( 0, 3 ) = 0.f;
        projM( 1, 0 ) = 0.f; projM( 1, 1 ) = 1.f / h; projM( 1, 2 ) = 0.f; projM( 1, 3 ) = 0.f;
        projM( 2, 0 ) = 0.f; projM( 2, 1 ) = 0.f;
        projM( 2, 2 ) = -2.f / ( params_.cameraDfar - params_.cameraDnear );
        projM( 2, 3 ) = -( params_.cameraDfar + params_.cameraDnear ) / ( params_.cameraDfar - params_.cameraDnear );
        projM( 3, 0 ) = 0.f; projM( 3, 1 ) = 0.f; projM( 3, 2 ) = 0.f; projM( 3, 3 ) = 1.f;
    }
    else
    {
        // setup perspective matrix
        float angle = 0.5f * params_.cameraViewAngle / 180.0f * PI_F;
        float h = tan( angle ) * params_.cameraDnear;
        float d = h * width( viewportRect_ ) / height( viewportRect_ );
        projM( 0, 0 ) = params_.cameraDnear / d; projM( 0, 1 ) = 0.f; projM( 0, 2 ) = 0.f; projM( 0, 3 ) = 0.f;
        projM( 1, 0 ) = 0.f; projM( 1, 1 ) = params_.cameraDnear / h; projM( 1, 2 ) = 0.f; projM( 1, 3 ) = 0.f;
        projM( 2, 0 ) = 0.f; projM( 2, 1 ) = 0.f;
        projM( 2, 2 ) = ( params_.cameraDfar + params_.cameraDnear ) / ( params_.cameraDnear - params_.cameraDfar );
        projM( 2, 3 ) = -2.f * ( params_.cameraDfar * params_.cameraDnear ) / ( params_.cameraDfar - params_.cameraDnear );
        projM( 3, 0 ) = 0.f; projM( 3, 1 ) = 0.f; projM( 3, 2 ) = -1.f; projM( 3, 3 ) = 0.f;
    }
}

void Viewport::setupStaticProjMatrix() const
{
    float h = (params_.cameraEye - params_.cameraCenter).length();
    float d = h * width( viewportRect_ ) / height( viewportRect_ );
    staticProj( 0, 0 ) = 1.f / d; staticProj( 0, 1 ) = 0.f; staticProj( 0, 2 ) = 0.f; staticProj( 0, 3 ) = 0.f;
    staticProj( 1, 0 ) = 0.f; staticProj( 1, 1 ) = 1.f / h; staticProj( 1, 2 ) = 0.f; staticProj( 1, 3 ) = 0.f;
    staticProj( 2, 0 ) = 0.f; staticProj( 2, 1 ) = 0.f;
    staticProj( 2, 2 ) = -2.f / (params_.cameraDfar - params_.cameraDnear);
    staticProj( 2, 3 ) = -(params_.cameraDfar + params_.cameraDnear) / (params_.cameraDfar - params_.cameraDnear);
    staticProj( 3, 0 ) = 0.f; staticProj( 3, 1 ) = 0.f; staticProj( 3, 2 ) = 0.f; staticProj( 3, 3 ) = 1.f;
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
        setRotationPivot_( sceneCenter );
    }

    auto sceneCenter = sceneBox_.valid() ? sceneBox_.center() : Vector3f();
    distToSceneCenter_ = ( getCameraPoint() - sceneCenter ).length();

    Vector3f coord = projectToViewportSpace( rotationPivot_ );
    static_viewport_point = Vector2f( coord[0], coord[1] );
    static_point_ = worldToCameraSpace( rotationPivot_ );
}

void Viewport::rotateView_() const
{
    // TODO: try to simplify
    AffineXf3f xf(viewM);
    Vector3f shift = static_point_ - xf.A * rotationPivot_;
    viewM.setTranslation( shift );

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

    // changing translation should not really be here, so const cast is OK
    // meanwhile it is because we need to keep distance(camera, scene center) static
    Vector3f* transConstCasted = const_cast<Vector3f*>( &params_.cameraTranslation );
    *transConstCasted = Matrix3f( params_.cameraTrackballAngle.inverse() ) * (shift + params_.cameraEye);
    *transConstCasted = params_.cameraTranslation / params_.cameraZoom;
    assert( !std::isnan( transConstCasted->x ) );
    viewM.setTranslation( shift );
}

void Viewport::transform_view( const AffineXf3f & xf )
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

const float Viewport::getPixelSize() const
{
    return ( tan( params_.cameraViewAngle * MR::PI_F / 360.0f ) * params_.cameraDnear * 2.0f ) / ( height( viewportRect_ ) * params_.cameraZoom );
}

void Viewport::setRotationPivot_( const Vector3f& point )
{
    rotationPivot_ = point;
}

// ================================================================
// projection part

const Matrix4f& Viewport::getViewMatrix() const
{
    return viewM;
}

AffineXf3f Viewport::getUnscaledViewXf() const
{
    // TODO. Try to find better normalize way
    AffineXf3f res( viewM );
    res.A.x = res.A.x.normalized();
    res.A.y = res.A.y.normalized();
    res.A.z = res.A.z.normalized();
    return res;
}

AffineXf3f Viewport::getViewXf() const
{
    return AffineXf3f( viewM );
}
const Matrix4f& Viewport::getProjMatrix() const
{
    return projM;
}
Matrix4f Viewport::getFullViewportMatrix() const
{
    return projM * viewM;
}
Matrix4f Viewport::getFullViewportInversedMatrix() const
{
    return (projM * viewM).inverse();
}

Vector3f Viewport::getUpDirection() const
{
    Vector3f res = Vector3f( viewM.y.x, viewM.y.y, viewM.y.z ).normalized();
    return res;
}
Vector3f Viewport::getRightDirection() const
{
    Vector3f res = Vector3f( viewM.x.x, viewM.x.y, viewM.x.z ).normalized();
    return res;
}
Vector3f Viewport::getBackwardDirection() const
{
    Vector3f res = Vector3f( viewM.z.x, viewM.z.y, viewM.z.z ).normalized();
    return res;
}

Line3f Viewport::unprojectPixelRay( const Vector2f& viewportPoint ) const
{
    auto M = getFullViewportInversedMatrix();
    Vector3f clipNear = viewportSpaceToClipSpace( Vector3f( viewportPoint.x, viewportPoint.y, 0.0f ) );
    auto clipFar = clipNear;
    clipFar.z = 1.0f;
    auto p = M( clipNear );
    auto d = M( clipFar ) - p;
    return Line3f( p, d );
}

Vector3f Viewport::worldToCameraSpace( const Vector3f& p ) const
{
    return viewM(p);
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

void Viewport::preciseFitDataToScreenBorder( const FitDataParams& fitParams )
{
    if ( fitParams.snapView )
        params_.cameraTrackballAngle = getClosestCanonicalQuaternion( params_.cameraTrackballAngle );

    const auto safeZoom = params_.cameraZoom;
    params_.cameraZoom = 1;

    Box3f box = calcBox_( fitParams.mode, params_.orthographic ? Space::CameraOrthographic : Space::World );
    if ( !box.valid() )
    {
        params_.cameraZoom = safeZoom;
        setRotationPivot_( Vector3f() );
        return;
    }

    auto dif = box.max - box.min;
    if ( params_.orthographic )
    {
        sceneBox_ = transformed( box, getViewXf_().inverse() );
    }
    else
    {
        sceneBox_ = box;
    }
    Vector3f sceneCenter = params_.orthographic ? 
        getViewXf_().inverse()( box.center() ) : box.center();
    setRotationPivot_( sceneCenter );

    params_.cameraTranslation = -sceneCenter;
    params_.cameraViewAngle = 45.0f;
    params_.objectScale = dif.length();
    if ( params_.objectScale == 0.0f )
        params_.objectScale = 1.0f;
    if ( params_.objectScale > cMaxObjectScale )
    {
        spdlog::warn( "Object scale exceeded its limit" );
        params_.objectScale = cMaxObjectScale;
    }

    if ( params_.orthographic )
    {
        auto factor = 1.f / (params_.cameraEye - params_.cameraCenter).length();
        auto tanFOV = tan(0.5f * params_.cameraViewAngle / 180.f * PI_F);
        params_.cameraZoom = factor / ( params_.objectScale * tanFOV );

        const auto winRatio = getRatio();
        auto dX = ( box.max.x - box.min.x ) / 2.f / winRatio;
        auto dY = ( box.max.y - box.min.y ) / 2.f;
        float maxD = std::max( dX, dY );

        if ( maxD == 0.0f )
            maxD = 1.0f;

        params_.cameraViewAngle = (2.f * atan2( maxD * params_.cameraZoom, params_.cameraDnear )) / PI_F * 180.f / fitParams.factor;
    }
    else
    {
        auto tanFOV = tan(0.5f * params_.cameraViewAngle / 180.f * PI_F);
        params_.cameraZoom = 1 / ( params_.objectScale * tanFOV );

        auto res = getZoomFOVtoScreen_( nullptr, fitParams.mode );
        if ( res.first == 0.0f )
            res.first = 1.0f;
        params_.cameraViewAngle = res.first / fitParams.factor;
    }

    needRedraw_ = true;
}

class LimitCalc
{
public:
    LimitCalc( const VertCoords & points, const VertBitSet & vregion, const std::function<bool(Vector3f&)> func ) 
        : points_( points ), vregion_( vregion ), func_(func) { }
    LimitCalc( LimitCalc& x, tbb::split ) 
        : points_( x.points_ ), vregion_( x.vregion_ ), func_(x.func_) { }
    void join(const LimitCalc& y) { box_.include(y.box_); }

    const Box3f& box() const { return box_; }

    void operator()(const tbb::blocked_range<VertId>& r)
    {
        for (VertId v = r.begin(); v < r.end(); ++v)
        {
            if ( !vregion_.test( v ) )
                continue;
            Vector3f pt = points_[v];
            if( func_( pt ) )
                box_.include( pt );
        }
    }

private:
    const VertCoords & points_;
    const VertBitSet & vregion_;
    const std::function<bool(Vector3f&)> func_;
    Box3f box_;
};

bool Viewport::allModelsInsideViewportRectangle() const
{
    auto res = getZoomFOVtoScreen_();
    return res.second && params_.cameraViewAngle > res.first;
}

Box3f Viewport::calcBox_( FitMode mode, Space space ) const
{
    Box3f box;
    const auto type = mode == FitMode::SelectedObjects ? ObjectSelectivityType::Selected : ObjectSelectivityType::Any;
    const auto allObj = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), type );

    const AffineXf3f xfV = getViewXf_();

    for( const auto& obj : allObj )
    {
        if( obj->globalVisibilty( id ) )
        {
            // object space to camera space
            auto xf = obj->worldXf();
            if ( space != Space::World )
                xf = xfV * xf;
            VertId lastValidVert;
            const VertCoords* coords = nullptr;
            const VertBitSet* selectedVerts = nullptr;
            auto objMesh = obj->asType<ObjectMeshHolder>();
            if ( objMesh )
            {
                if ( !objMesh->mesh() )
                    continue;
                const auto& mesh = *objMesh->mesh();
                lastValidVert = mesh.topology.lastValidVert();
                coords = &mesh.points;
                selectedVerts = &mesh.topology.getValidVerts();
            }
            else if ( auto objLines = obj->asType<ObjectLinesHolder>() )
            {
                if ( !objLines->polyline() )
                    continue;
                const auto& polyline = *objLines->polyline();
                lastValidVert = polyline.topology.lastValidVert();
                coords = &polyline.points;
                selectedVerts = &polyline.topology.getValidVerts();
            }
            else if ( auto objPoints = obj->asType<ObjectPointsHolder>() )
            {
                if ( !objPoints->pointCloud() )
                    continue;
                const auto& pointCloud = *objPoints->pointCloud();
                lastValidVert = VertId( pointCloud.validPoints.size() - 1 );// TODO: last valid
                coords = &pointCloud.points;
                selectedVerts = &pointCloud.validPoints;
            }
            else if ( obj->asType<ObjectLabel>() )
            {
                // do nothing
            }
            else
            {
                assert( false );
                continue;
            }
            VertBitSet myVerts;
            if( mode == FitMode::SelectedPrimitives )
            {
                selectedVerts = nullptr;
                if ( objMesh )
                {
                    myVerts = getIncidentVerts( objMesh->mesh()->topology, objMesh->getSelectedEdges() ) | 
                        getIncidentVerts( objMesh->mesh()->topology, objMesh->getSelectedFaces() );
                    if ( !myVerts.any() )
                        continue;
                    selectedVerts = &myVerts;
                }
            }
            if ( !selectedVerts )
                continue;
            std::function<bool( Vector3f& )> func;
            if( space == Space::CameraOrthographic || space == Space::World )
            {
                func = [&]( Vector3f& p ) 
                {
                    p = xf( p );
                    return true;
                };
            }
            else
            {
                func = [&]( Vector3f& p ) 
                {
                    auto v = xf( p );
                    if ( v.z == 0 )
                        return false;
                    p = Vector3f( v.x / v.z, v.y / v.z, v.z );
                    return true;
                };
            }
            LimitCalc calc( *coords, *selectedVerts, func );
            parallel_reduce( tbb::blocked_range<VertId>( VertId{ 0 }, lastValidVert + 1 ), calc );
            box.include( calc.box() );
        }
    }
    return box;
}

std::pair<float, bool> Viewport::getZoomFOVtoScreen_( Vector3f* cameraShift, FitMode mode ) const
{
    const auto box = calcBox_( mode, params_.orthographic ? Space::CameraOrthographic : Space::CameraPerspective );
    if ( !box.valid() )
        return std::make_pair( params_.cameraViewAngle, true );

    // z points from scene to camera
    bool allInside = (-box.max.z < params_.cameraDfar) && (-box.min.z > params_.cameraDnear);

    const auto winRatio = getRatio();
    if( params_.orthographic )
    {
        auto dX = (box.max.x - box.min.x) / 2.f / winRatio;
        auto dY = (box.max.y - box.min.y) / 2.f;
        float maxD = std::max(dX, dY);
        if( cameraShift )
        {
            auto meanX = (box.max.x + box.min.x) / 2.f;
            auto meanY = (box.max.y + box.min.y) / 2.f;
            const AffineXf3f xfV = getViewXf_();
            *cameraShift = -xfV.A.x.normalized() * (meanX / params_.cameraZoom) - xfV.A.y.normalized() * (meanY / params_.cameraZoom);
        }
        return std::make_pair( (2.f * atan2( maxD, params_.cameraDnear )) / PI_F * 180.f, allInside );
    }
    else
    {
        auto maxX = std::max( box.max.x, -box.min.x ) / winRatio;
        auto maxY = std::max( box.max.y, -box.min.y );
        return std::make_pair( (2.f * atan( std::max( maxX, maxY ) ) ) / PI_F * 180.f, allInside );
    }
}

void Viewport::fitData( float fill, bool snapView )
{
    updateSceneBox_();

    if ( !sceneBox_.valid() )
    {
        setRotationPivot_( Vector3f() );
        return;
    }

    auto sceneCenter = sceneBox_.center();
    setRotationPivot_( sceneCenter );
    params_.cameraTranslation = -sceneCenter;

    auto dif = sceneBox_.max - sceneBox_.min;
    params_.cameraViewAngle = 45.0f;
    params_.objectScale = dif.length();
    if ( params_.objectScale == 0.0f )
        params_.objectScale = 1.0f;
    if ( params_.objectScale > cMaxObjectScale )
    {
        spdlog::warn( "Object scale exceeded its limit" );
        params_.objectScale = cMaxObjectScale;
    }

    auto tanFOV = tan(0.5f * params_.cameraViewAngle / 180.f * PI_F);
    auto factor = params_.orthographic ? 1.f / (params_.cameraEye - params_.cameraCenter).length() : 1.f;
    params_.cameraZoom = factor * fill / ( params_.objectScale * tanFOV );

    if ( snapView )
        params_.cameraTrackballAngle = getClosestCanonicalQuaternion( params_.cameraTrackballAngle );

    needRedraw_ = true;
}

// ================================================================
// set-get parameters part

float Viewport::getRatio() const
{
    // tan([left-right] / 2) / tan([up-down] / 2)
    return width( viewportRect_ ) / height( viewportRect_ );
}

Vector3f Viewport::getCameraPoint() const
{
    AffineXf3f xfInv = AffineXf3f( viewM ).inverse();
    return xfInv.b;
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

    Vector3f lookDir = params_.cameraCenter - params_.cameraEye; 
    auto rotLook = Matrix3f::rotation( newDir, lookDir );

    auto upVec = rotLook.inverse() * params_.cameraUp;
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

}
