#include "MRViewport.h"
#include "MRViewer.h"
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRArrow.h>
#include <MRMesh/MRUVSphere.h>
#include <MRMesh/MRToFromEigen.h>
#include <MRMesh/MRClosestPointInTriangle.h>
#include <MRMesh/MRTimer.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRGLMacro.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRGLStaticHolder.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPolylineProject.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRPch/MRTBB.h"
#include "MRMesh/MR2to3.h"

using VisualObjectTreeDataVector = std::vector<MR::VisualObject*>;
namespace
{
void getPickerDataVector( MR::Object& obj, MR::ViewportMask id, VisualObjectTreeDataVector& outVector )
{
    if ( !obj.isVisible( id ) )
        return;
    if ( auto visobj = obj.asType<MR::VisualObject>() )
        if ( visobj->isPickable( id ) )
            outVector.push_back( {visobj} );
    for ( const auto& child : obj.children() )
        getPickerDataVector( *child, id, outVector );
}
}

namespace MR
{

Viewport::Viewport()
{
}

Viewport::~Viewport()
{
}

void Viewport::init()
{
    viewportGL_ = ViewportGL();
    viewportGL_.init();
    init_axes();
    updateSceneBox_();
    setRotationPivot_( sceneBox_.valid() ? sceneBox_.center() : Vector3f() );
    setupProjMatrix();
    setupStaticProjMatrix();
}

void Viewport::shut()
{
    viewportGL_.free();
}

// ================================================================
// draw functions part

void Viewport::draw(const VisualObject& obj, const AffineXf3f& xf, bool forceZBuffer, bool alphaSort ) const
{
    auto modelTemp = Matrix4f( xf );
    auto normTemp = viewM * modelTemp;
    if ( normTemp.det() == 0 )
    {
        auto norm = normTemp.norm();
        if ( std::isnormal( norm ) )
        {
            normTemp /= norm;
            normTemp.w = { 0, 0, 0, 1 };
        }
        else
        {
            spdlog::warn( "Object transform is degenerate" );
            return;
        }
    }
    auto normM = normTemp.inverse().transposed();

    RenderParams params
    {
        {viewM.data(), modelTemp.data(), projM.data(), normM.data(),
        id, params_.clippingPlane, toVec4<int>( viewportRect_ )},
        params_.lightPosition, forceZBuffer, alphaSort
    };
    obj.render( params );
}

void Viewport::clear_framebuffers() const
{
    viewportGL_.fillViewport( toVec4<int>( viewportRect_ ), params_.backgroundColor );
}

ObjAndPick Viewport::pick_render_object() const
{
    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    return pick_render_object( renderVector );
}

ObjAndPick Viewport::pick_render_object( const Vector2f& viewportPoint ) const
{
    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    return pick_render_object( renderVector, viewportPoint );
}

ObjAndPick Viewport::pick_render_object( const std::vector<VisualObject*>& renderVector ) const
{
    auto& viewer = getViewerInstance();
    const auto& mousePos = viewer.mouseController.getMousePos();
    auto viewerPoint = viewer.screenToViewport(
        Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), id );
    return pick_render_object( renderVector, Vector2f( viewerPoint.x, viewerPoint.y ) );
}

ObjAndPick Viewport::pick_render_object( const std::vector<VisualObject*>& renderVector, const Vector2f& viewportPoint ) const
{
    return multiPickObjects( renderVector, {viewportPoint} ).front();
}

std::vector<ObjAndPick> Viewport::multiPickObjects( const std::vector<VisualObject*>& renderVector, const std::vector<Vector2f>& viewportPoints ) const
{
    MR_TIMER;
    if ( viewportPoints.empty() )
        return {};
    std::vector<Vector2i> picks( viewportPoints.size() );
    ViewportGL::PickParameters params{
        renderVector,
        {viewM.data(),projM.data(),toVec4<int>( viewportRect_ )},
        params_.clippingPlane,id};

    for ( int i = 0; i < viewportPoints.size(); ++i )
        picks[i] = Vector2i( viewportPoints[i] );

    std::vector<ObjAndPick> result( picks.size() );

    if ( width( viewportRect_ ) == 0 || height( viewportRect_ ) == 0 )
        return result;

    auto pickResult = viewportGL_.pickObjects( params, picks );
    for ( int i = 0; i < pickResult.size(); ++i )
    {
        auto& pickRes = pickResult[i];
        if ( pickRes.geomId == -1 || pickRes.primId == -1 )
            continue;

        PointOnFace res;
        res.face = FaceId( int( pickRes.primId ) );
        if ( auto pointObj = renderVector[pickRes.geomId]->asType<ObjectPointsHolder>() )
        {
            if ( auto pc = pointObj->pointCloud() )
            {
                auto vid = VertId( int( pickRes.primId ) );
                if ( !pc->validPoints.test( vid ) )
                    continue;
                res.point = pc->points[vid];
            }
            else
            {
                res.point = renderVector[pickRes.geomId]->worldXf().inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
            }
        }
        else if ( auto linesObj = renderVector[pickRes.geomId]->asType<ObjectLinesHolder>() )
        {
            res.point = renderVector[pickRes.geomId]->worldXf().inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
            UndirectedEdgeId ue{ int( pickRes.primId ) };
            if ( auto pl = linesObj->polyline() )
                res.point = closestPointOnLineSegm( res.point, { pl->orgPnt( ue ), pl->destPnt( ue ) } );
        }
        else if ( auto meshObj = renderVector[pickRes.geomId]->asType<ObjectMeshHolder>() )
        {
            if ( res.face.valid() )
            {
                const auto& mesh = meshObj->mesh();
                if ( mesh && !mesh->topology.hasFace( res.face ) )
                {
                    assert( false );
                    continue;
                }

                res.point = renderVector[pickRes.geomId]->worldXf().inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
                if ( mesh )
                {
                    Vector3f a, b, c;
                    mesh->getTriPoints( res.face, a, b, c );
                    res.point = closestPointInTriangle( res.point, a, b, c ).first;
                }
            }
        }
        if ( auto parent = renderVector[pickRes.geomId]->parent() )
        {
            for ( auto& child : parent->children() )
                if ( child.get() == renderVector[pickRes.geomId] )
                {
                    result[i] = {std::dynamic_pointer_cast<VisualObject>( child ),res};
                    continue;
                }
        }
        else
        {
            // object is not in scene
            assert( false );
            continue;
        }
    }
    return result;
}

std::vector<std::shared_ptr<MR::VisualObject>> Viewport::findObjectsInRect( const Box2i& rect,
                                                                            int maxRenderResolutionSide ) const
{
    MR_TIMER;

    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    ViewportGL::PickParameters params{
        renderVector,
        {viewM.data(),projM.data(),toVec4<int>( viewportRect_ )},
        params_.clippingPlane,id };

    auto viewportRect = Box2i( Vector2i( 0, 0 ), Vector2i( int( width( viewportRect_ ) ), int( height( viewportRect_ ) ) ) );
    auto pickResult = viewportGL_.findUniqueObjectsInRect( params, rect.intersection( viewportRect ), maxRenderResolutionSide );
    std::vector<std::shared_ptr<VisualObject>> result( pickResult.size() );
    for ( int i = 0; i < pickResult.size(); ++i )
    {
        if ( auto parent = renderVector[pickResult[i]]->parent() )
        {
            for ( auto& child : parent->children() )
            {
                if ( child.get() == renderVector[pickResult[i]] )
                {
                    result[i] = std::dynamic_pointer_cast< VisualObject >( child );
                }
            }
        }
    }

    return result;
}

ConstObjAndPick Viewport::const_pick_render_object() const
{
    return pick_render_object();
}

ConstObjAndPick Viewport::const_pick_render_object( const std::vector<const VisualObject*>& objects ) const
{
    // not to duplicate code
    return pick_render_object( reinterpret_cast<const std::vector<VisualObject*>&> ( objects ) );
}

std::vector<ConstObjAndPick> Viewport::constMultiPickObjects( const std::vector<const VisualObject*>& objects, const std::vector<Vector2f>& viewportPoints ) const
{
    auto pickRes = multiPickObjects( reinterpret_cast<const std::vector<VisualObject*>&> ( objects ), viewportPoints );
    std::vector<ConstObjAndPick> res( pickRes.size() );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = pickRes[i];
    return res;
}

const ViewportPointsWithColors& Viewport::getPointsWithColors() const
{
    return viewportGL_.getPointsWithColors();
}

const ViewportLinesWithColors& Viewport::getLinesWithColors() const
{
    return viewportGL_.getLinesWithColors();
}

void Viewport::setPointsWithColors( const ViewportPointsWithColors& pointsWithColors )
{
    if ( beforeSetPointsWithColors )
        beforeSetPointsWithColors( getPointsWithColors(), pointsWithColors );
    viewportGL_.setPointsWithColors( pointsWithColors );
}

void Viewport::setLinesWithColors( const ViewportLinesWithColors& linesWithColors )
{
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    if ( beforeSetLinesWithColors )
        beforeSetLinesWithColors( getLinesWithColors(), linesWithColors );
    MR_SUPPRESS_WARNING_POP
    viewportGL_.setLinesWithColors( linesWithColors );
}

void Viewport::setupView() const
{
    setupViewMatrix();
    setupProjMatrix();
    setupStaticProjMatrix();
}

void Viewport::preDraw() const
{
    if( previewLinesDepthTest_ )
        draw_lines();
    if( previewPointsDepthTest_ )
        draw_points();
    draw_rotation_center();
    draw_global_basis();
}

void Viewport::postDraw() const
{
    draw_border();
    draw_clipping_plane();

    if( !previewLinesDepthTest_ )
        draw_lines();
    if( !previewPointsDepthTest_ )
        draw_points();

    // important to be last
    draw_axes();
}

void Viewport::updateSceneBox_()
{
    sceneBox_ = SceneRoot::get().getWorldTreeBox( id );
}

void Viewport::setViewportRect( const Viewport::ViewportRectangle& rect )
{
    if ( rect == viewportRect_ )
        return;
    needRedraw_ = true;
    viewportRect_ = rect;
    init_axes();
}

const Viewport::ViewportRectangle& Viewport::getViewportRect() const
{
    return viewportRect_;
}

// ================================================================
// projection part

const Box3f& Viewport::getSceneBox() const
{
    return sceneBox_;
}

void Viewport::setBackgroundColor( const Color& color )
{
    if ( params_.backgroundColor == color )
        return;
    params_.backgroundColor = color; 
    needRedraw_ = true;
}

void Viewport::setClippingPlane( const Plane3f& plane )
{
    if ( params_.clippingPlane == plane )
        return;
    params_.clippingPlane = plane; 
    needRedraw_ = true;
}

void Viewport::showAxes( bool on )
{
    Viewer::constInstance()->basisAxes->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->basisAxes->getRedrawFlag( id );
    Viewer::constInstance()->basisAxes->resetRedrawFlag();
}

void Viewport::showClippingPlane( bool on )
{
    Viewer::constInstance()->clippingPlaneObject->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->clippingPlaneObject->getRedrawFlag( id );
    Viewer::constInstance()->clippingPlaneObject->resetRedrawFlag();
}

void Viewport::showRotationCenter( bool on )
{
    Viewer::constInstance()->rotationSphere->setVisible( on, id );
}

void Viewport::rotationCenterMode( Parameters::RotationCenterMode mode )
{
    if ( mode == params_.rotationMode )
        return;
    params_.rotationMode = mode;
    needRedraw_ = true;
}

void Viewport::showGlobalBasis( bool on )
{
    Viewer::constInstance()->globalBasisAxes->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->globalBasisAxes->getRedrawFlag( id );
    Viewer::constInstance()->globalBasisAxes->resetRedrawFlag();
}

void Viewport::setParameters( const Viewport::Parameters& params )
{
    if ( params == params_ )
        return;
    params_ = params;
    needRedraw_ = true;
}

void Viewport::set_axes_size( const int axisPixSize )
{
    if ( axisPixSize == axisPixSize_ )
        return;
    needRedraw_ = true;
    axisPixSize_ = axisPixSize;
    init_axes();
}

void Viewport::set_axes_pose( const int pixelXoffset, const int pixelYoffset )
{
    if ( pixelXoffset_ == pixelXoffset &&
         pixelYoffset_ == pixelYoffset )
        return;
    needRedraw_ = true;
    pixelXoffset_ = pixelXoffset;
    pixelYoffset_ = pixelYoffset;
    init_axes();
}

// ================================================================
// GL functions part

void Viewport::draw_points( void ) const
{
    ViewportGL::RenderParams params{getBaseRenderParams()};
    params.cameraZoom = params_.cameraZoom;
    params.zOffset = pointsZoffset;
    params.depthTest = previewPointsDepthTest_;
    params.width = point_size;

    viewportGL_.drawPoints( params );
}

void Viewport::draw_border() const
{    
    viewportGL_.drawBorder( getBaseRenderParams(), params_.borderColor );
}

// ================================================================
// additional elements

void Viewport::init_axes()
{
    // find relative points for axes
    float axesX, axesY;
    if(pixelXoffset_ < 0)
        axesX = width( viewportRect_ ) + pixelXoffset_;
    else
        axesX = float(pixelXoffset_);
    if(pixelYoffset_ < 0)
        axesY = height( viewportRect_ ) + pixelYoffset_;
    else
        axesY = float(pixelYoffset_);
    const float pixSize = float(axisPixSize_) / sqrtf(2);
    relPoseBase = { axesX, axesY, 0.5f };
    relPoseSide = { axesX + pixSize, axesY + pixSize, 0.5f };
}

void Viewport::draw_axes() const
{
    if ( Viewer::constInstance()->basisAxes->isVisible( id ) )
    {
        auto fullInversedM = (staticProj * viewM).inverse();
        auto transBase = fullInversedM( viewportSpaceToClipSpace( relPoseBase ) );
        auto transSide = fullInversedM( viewportSpaceToClipSpace( relPoseSide ) );

        float scale = (transSide - transBase).length();
        params_.basisAxesXf = AffineXf3f( Matrix3f::scale( scale ), transBase );
        std::swap( staticProj, projM );
        draw( *Viewer::constInstance()->basisAxes, params_.basisAxesXf, true );
        draw( *Viewer::constInstance()->basisAxes, params_.basisAxesXf );
        for ( const auto& child : getViewerInstance().basisAxes->children() )
        {
            if ( auto visualChild = child->asType<VisualObject>() )
                draw( *visualChild, params_.basisAxesXf );
        }
        std::swap( staticProj, projM );
    }
}

void Viewport::draw_clipping_plane() const
{
    const auto& v = Viewer::constInstance();
    if ( !v->clippingPlaneObject->isVisible( id ) )
        return;

    AffineXf3f transform = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), params_.clippingPlane.n ) );
    transform = AffineXf3f::linear( Matrix3f::scale( params_.cameraDfar - params_.cameraDnear )/ params_.cameraZoom ) * transform;
    transform.b = params_.clippingPlane.n * params_.clippingPlane.d;
    draw( *Viewer::constInstance()->clippingPlaneObject, transform );
}

void Viewport::draw_global_basis() const
{
    if ( !Viewer::instance()->globalBasisAxes->isVisible( id ) )
        return;

    params_.globalBasisAxesXf = AffineXf3f::linear( Matrix3f::scale( params_.objectScale * 0.5f ) );
    draw( *Viewer::constInstance()->globalBasisAxes, params_.globalBasisAxesXf );
}

void Viewport::draw_lines( void ) const
{
    ViewportGL::RenderParams params{getBaseRenderParams()};
    params.cameraZoom = params_.cameraZoom;
    params.zOffset = linesZoffset;
    params.depthTest = previewLinesDepthTest_;
    params.width = line_width;

    viewportGL_.drawLines( params );
}

void  Viewport::add_line( const Vector3f& start_pos, const Vector3f& fin_pos,
                                 const Color& color_start, const Color& color_fin )
{
    auto [newLines, newColors] = viewportGL_.getLinesWithColors();
    newLines.push_back( { start_pos, fin_pos } );
    newColors.push_back( { Vector4f( color_start ),Vector4f( color_fin ) } );
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    setLinesWithColors( { newLines,newColors } );
    MR_SUPPRESS_WARNING_POP
    needRedraw_ = viewportGL_.lines_dirty;
}

void Viewport::add_lines( const std::vector<Vector3f>& points, const std::vector<Color>& colorsArg )
{
    if ( points.size() < 2 )
        return;
    auto lc = viewportGL_.getLinesWithColors();
    auto& newLines = lc.lines;
    auto& newColors = lc.colors;

    auto oldSize = newLines.size();
    newLines.resize( oldSize + ( points.size() - 1 ) );
    newColors.resize( newLines.size() );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() - 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            const auto& start_pos = points[i];
            const auto& fin_pos = points[i + 1];
            auto ind = oldSize + i;

            newLines[ind] = LineSegm3f{ start_pos, fin_pos };
            newColors[ind] = { Vector4f( colorsArg[i] ),Vector4f( colorsArg[i + 1] ) };
        }
    } );
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    setLinesWithColors( { newLines,newColors } );
    MR_SUPPRESS_WARNING_POP
    needRedraw_ = viewportGL_.lines_dirty;
}

void Viewport::add_lines( const std::vector<Vector3f>& points, const Color& color )
{
    std::vector<Color> colors( points.size(), color );
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    add_lines( points, colors );
    MR_SUPPRESS_WARNING_POP
}

void  Viewport::remove_lines(  )
{
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    setLinesWithColors( { {},{} } );
    MR_SUPPRESS_WARNING_POP
    needRedraw_ = viewportGL_.lines_dirty;
}

void  Viewport::add_point ( const Vector3f& pos, const Color& color )
{
    auto [newPoints, newColors] = viewportGL_.getPointsWithColors();
    newPoints.push_back( pos );
    newColors.push_back( Vector4f( color ) );
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 );
    setPointsWithColors( { newPoints,newColors } );
    MR_SUPPRESS_WARNING_POP
    needRedraw_ = viewportGL_.points_dirty;
} 

void  Viewport::remove_points()
{
    MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    setPointsWithColors( { {},{} } );
    MR_SUPPRESS_WARNING_POP
    needRedraw_ = viewportGL_.points_dirty;
}

void Viewport::setPreviewLinesDepthTest( bool on )
{
    if ( previewLinesDepthTest_ == on )
        return;
    previewLinesDepthTest_ = on;
    needRedraw_ = true;
}

void Viewport::setPreviewPointsDepthTest( bool on )
{
    if ( previewPointsDepthTest_ == on )
        return;
    previewPointsDepthTest_ = on;
    needRedraw_ = true;
}

bool Viewport::Parameters::operator==( const Viewport::Parameters& other ) const
{
    return
        backgroundColor == other.backgroundColor &&
        lightPosition == other.lightPosition &&
        cameraTrackballAngle == other.cameraTrackballAngle &&
        cameraTranslation == other.cameraTranslation &&
        cameraZoom == other.cameraZoom &&
        cameraViewAngle == other.cameraViewAngle &&
        cameraDnear == other.cameraDnear &&
        cameraDfar == other.cameraDfar &&
        depthTest == other.depthTest &&
        orthographic == other.orthographic &&
        objectScale == objectScale &&
        borderColor == other.borderColor &&
        clippingPlane == other.clippingPlane &&
        rotationMode == other.rotationMode &&
        selectable == other.selectable;
}

}
