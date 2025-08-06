#include "MRSurfaceManipulationWidget.h"
#include "MRMouseController.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRAppendHistory.h"
#include "MRMouse.h"
#include "MRPalette.h"
#include "MRProjectMeshAttributes.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRSurfaceDistance.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MREnumNeighbours.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRFillHoleNicely.h"
#include "MRMesh/MRLaplacian.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRChangeMeshAction.h"
#include "MRMesh/MRPartialChangeMeshAction.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRChangeSelectionAction.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRSceneCache.h"
#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRPointsProject.h"

namespace MR
{

void findSpaceDistancesAndVerts( const Mesh& mesh, const VertBitSet& start, float range, VertScalars& distances, VertBitSet& verts, bool codirected, const VertBitSet* untouchable )
{
    // note! better update bitset for interesting verts than check for all verts from distances
    MR_TIMER;
    auto tree = AABBTreePoints( mesh.points, start );

    EnumNeihbourVertices e;
    e.run( mesh.topology, start, [&] ( VertId v )
    {
        const auto proj = findProjectionOnPoints( mesh.points[v], tree );
        const float dist = std::sqrt( proj.distSq );
        const bool valid = ( dist <= range ) && ( !codirected || ( dot( mesh.normal( v ), mesh.normal( proj.vId ) ) >= 0.f ) );
        verts.set( v, valid );
        bool canChange = !untouchable || dist < distances[v] || !untouchable->test( v );
        if ( canChange )
            distances[v] = dist;
        return valid;
    } );
}

/// Undo action for ObjectMesh points only (not topology) change;
/// It can store all points (uncompressed format), or only modified points (compressed format)
class SurfaceManipulationWidget::SmartChangeMeshPointsAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's mesh points in uncompressed format before making any changes in it
    SmartChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        stdAction_{ std::make_unique<ChangeMeshPointsAction>( std::move( name ), obj ) }
    {
    }

    virtual std::string name() const override
    {
        return stdAction_ ? stdAction_->name() : diffAction_->name();
    }

    virtual void action( HistoryAction::Type t ) override
    {
        if ( stdAction_ )
            stdAction_->action( t );
        else
            diffAction_->action( t );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return MR::heapBytes( stdAction_ ) + MR::heapBytes( diffAction_ );
    }

    /// switch from uncompressed to compressed format to occupy less amount of memory
    void compress()
    {
        assert( stdAction_ );
        if ( stdAction_ )
        {
            diffAction_ = std::make_unique<PartialChangeMeshPointsAction>(
                stdAction_->name(), stdAction_->obj(), cmpOld, stdAction_->clonePoints() );
            stdAction_.reset();
        }
        assert( !stdAction_ );
        assert( diffAction_ );
    }

private:
    std::unique_ptr<ChangeMeshPointsAction> stdAction_;
    std::unique_ptr<PartialChangeMeshPointsAction> diffAction_;
};


//const float k = r < 1-a ? std::sqrt( sqr( 1 - a ) - sqr( r ) ) + ( 1 - a ) : -std::sqrt( sqr( a ) - sqr( r - 1 ) ) + a; // alternative version F_point_shift(r,i) (i == a)

// not in the header to be able to destroy Laplacian
SurfaceManipulationWidget::SurfaceManipulationWidget()
{
}

// not in the header to be able to destroy Laplacian
SurfaceManipulationWidget::~SurfaceManipulationWidget()
{
}

void SurfaceManipulationWidget::init( const std::shared_ptr<ObjectMesh>& objectMesh )
{
    assert( objectMesh );
    obj_ = objectMesh;

    if ( firstInit_ )
    {
        float diagonal = obj_->getBoundingBox().diagonal();
        settings_.radius = diagonal * 0.02f;
        settings_.relaxForce = 0.2f;
        settings_.editForce = diagonal * 0.01f;
        settings_.relaxForceAfterEdit = 0.25f;
        settings_.workMode = WorkMode::Add;
        firstInit_ = false;
    }
    if ( !palette_ )
    {
        palette_ = std::make_shared<Palette>( Palette::DefaultColors );
        palette_->setFilterType( FilterType::Linear );
    }

    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;

    if ( !originalMesh_ )
    {
        originalMesh_ = std::make_shared<Mesh>( *obj_->mesh() );

        const float rangeLength = settings_.editForce * ( Palette::DefaultColors.size() - 1 );
        palette_->setRangeMinMax( rangeLength * -0.5f, rangeLength * 0.5f );

        valueChanges_.clear();
        obj_->setAncillaryUVCoords( VertUVCoords( numV, { 0.5f, 1.f } ) );
    }

    reallocData_( numV );

    updateTexture();

    initConnections_();

    mousePressed_ = false;
    mousePos_ = { -1, -1 };

    sameValidVerticesAsInOriginMesh_ = true;
}

void SurfaceManipulationWidget::reset()
{
    originalMesh_.reset();

    removeLastStableObjMesh_();

    obj_->clearAncillaryTexture();
    obj_.reset();

    clearData_();

    resetConnections_();
    mousePressed_ = false;
}

void SurfaceManipulationWidget::setFixedRegion( const FaceBitSet& region )
{
    unchangeableVerts_ = getIncidentVerts( obj_->mesh()->topology, region ) ;
}

void SurfaceManipulationWidget::setSettings( const Settings& settings )
{
    if ( mousePressed_ )
        return;

    settings_ = settings;
    settings_.radius = std::max( settings_.radius, 1.e-5f );
    settings_.relaxForce = std::clamp( settings_.relaxForce, 0.001f, 0.5f );
    settings_.editForce = std::max( settings_.editForce, 1.e-5f );
    settings_.relaxForceAfterEdit = std::clamp( settings_.relaxForceAfterEdit, 0.f, 0.5f );
    settings_.sharpness = std::clamp( settings_.sharpness, 0.f, 100.f );
    updateRegion_( mousePos_ );
}

void SurfaceManipulationWidget::updateTexture()
{
    MeshTexture texture;
    if ( enableDeviationTexture_ )
    {
        if ( palette_ )
        {
            MeshTexture palleteTexture = palette_->getTexture();
            texture.filter = palleteTexture.filter;
            texture.resolution = { palleteTexture.resolution.x, 2 };
            texture.pixels.resize( texture.resolution.x * texture.resolution.y );
            for ( int x = 0; x < palleteTexture.resolution.x; ++x )
            {
                texture.pixels[x] = Color( 255, 64, 64, 255 );
                texture.pixels[x + palleteTexture.resolution.x] = palleteTexture.pixels[x];
            }
        }
        else
        {
            texture.pixels = { Color( 255, 64, 64, 255 ), Color( 255, 64, 64, 255 ), Color( 255, 64, 64, 255 ),
                Color::blue(), Color::green(), Color::red() };
            texture.resolution = { 2, 2 };
        }
    }
    else
    {
        texture.pixels = { Color( 255, 64, 64, 255 ), Color( 0, 0, 0, 0 ) };
        texture.resolution = { 1, 2 };
    }
    obj_->setAncillaryTexture( texture );
}

void SurfaceManipulationWidget::updateUVs()
{
    updateRegionUVs_( obj_->mesh()->topology.getValidVerts() );
}

void SurfaceManipulationWidget::enableDeviationVisualization( bool enable )
{
    if ( enableDeviationTexture_ == enable )
        return;
    enableDeviationTexture_ = enable;
    updateTexture();
    updateUVs();
}

void SurfaceManipulationWidget::setDeviationCalculationMethod( DeviationCalculationMethod method )
{
    if ( sameValidVerticesAsInOriginMesh_ )
        deviationCalculationMethod_ = method;
    else
        deviationCalculationMethod_ = DeviationCalculationMethod::ExactDistance;
    updateValueChanges_( obj_->mesh()->topology.getValidVerts() );
}

Vector2f SurfaceManipulationWidget::getMinMax()
{
    const float rangeLength = settings_.editForce * ( Palette::DefaultColors.size() - 1 );
    return { rangeLength * -0.5f, rangeLength * 0.5f };
}

void SurfaceManipulationWidget::createLastStableObjMesh_()
{
    assert( !lastStableObjMesh_ );
    lastStableObjMesh_ = std::dynamic_pointer_cast< ObjectMesh >( obj_->clone() );
    lastStableObjMesh_->setAncillary( true );
    lastStableObjMesh_->setVisible( false );
    obj_->setPickable( false );
    obj_->parent()->addChild( lastStableObjMesh_ );
}

void SurfaceManipulationWidget::removeLastStableObjMesh_()
{
    if ( lastStableObjMesh_ )
    {
        lastStableObjMesh_->detachFromParent();
        lastStableObjMesh_.reset();
    }
    obj_->setPickable( true );
}

bool SurfaceManipulationWidget::onMouseDown_( MouseButton button, int modifiers )
{
    if ( button != MouseButton::Left || !checkModifiers_( modifiers ) )
        return false;

    ObjAndPick objAndPick;
    if ( ignoreOcclusion_ )
        objAndPick = getViewerInstance().viewport().pickRenderObject( { { static_cast< VisualObject* >( obj_.get() ) } } );
    else
        objAndPick = getViewerInstance().viewport().pick_render_object();
    if ( !objAndPick.first || objAndPick.first != obj_ )
        return false;

    mousePressed_ = true;
    if ( settings_.workMode == WorkMode::Laplacian )
    {
        if ( !objAndPick.second.face.valid() )
            return false;

        if ( badRegion_ )
        {
            mousePressed_ = false;
            return false;
        }
        laplacianPickVert_( objAndPick.second );
    }
    else
    {
        if ( settings_.workMode != WorkMode::Patch )
        {
            // in patch mode the mesh does not change till mouse up, and we always need to pick in it (before and right after patch)
            createLastStableObjMesh_();
            lastStableValueChanges_ = valueChanges_;

            appendHistoryAction_ = true;
            std::string name = "Brush: ";
            if ( settings_.workMode == WorkMode::Add )
                name += "Add";
            else if ( settings_.workMode == WorkMode::Remove )
                name += "Remove";
            else if ( settings_.workMode == WorkMode::Relax )
                name += "Smooth";

            historyAction_ = std::make_shared<SmartChangeMeshPointsAction>( name, obj_ );
        }
        changeSurface_();
    }

    return true;
}

void SurfaceManipulationWidget::compressChangePointsAction_()
{
    if ( historyAction_ )
    {
        historyAction_->compress();
        historyAction_.reset();
    }
}

void SurfaceManipulationWidget::updateDistancesAndRegion_( const Mesh& mesh, const VertBitSet& start, VertScalars& distances, VertBitSet& region, const VertBitSet* untouchable )
{
    findSpaceDistancesAndVerts( mesh, start, settings_.radius, distances, region, editOnlyCodirectedSurface_, untouchable );
}

bool SurfaceManipulationWidget::onMouseUp_( Viewer::MouseButton button, int /*modifiers*/ )
{
    if ( button != MouseButton::Left || !mousePressed_ )
        return false;

    MR_FINALLY{ compressChangePointsAction_(); };

    mousePressed_ = false;
    if ( settings_.workMode == WorkMode::Laplacian )
    {
        removeLastStableObjMesh_();
        return true;
    }

    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    pointsShift_.clear();
    pointsShift_.resize( numV, 0.f );

    const auto & oldMesh = *obj_->varMesh();
    if ( settings_.workMode == WorkMode::Patch )
    {
        {
            // clear UV
            visualizationRegion_ |= generalEditingRegion_;
            expand( oldMesh.topology, visualizationRegion_, 2 );
            updateUVmap_( false );
        }
        const auto delFaces = getIncidentFaces( oldMesh.topology, generalEditingRegion_ );
        if ( delFaces.any() )
        {
            SCOPED_HISTORY( "Brush: Patch" );
            ownMeshChangedSignal_ = true;
            std::shared_ptr<Mesh> newMesh = std::make_shared<Mesh>( oldMesh );
            FaceBitSet newFaceSelection = obj_->getSelectedFaces() - delFaces;
            UndirectedEdgeBitSet newEdgeSelection = obj_->getSelectedEdges() - getInnerEdges( oldMesh.topology, delFaces ); // must be done before actual deletion
            auto bds = delRegionKeepBd( *newMesh, delFaces );
            const FaceBitSet oldFaces = newMesh->topology.getValidFaces();
            for ( const auto & bd : bds )
            {
                if ( bd.empty() )
                    continue;
                // assert( isHoleBd( mesh.topology, bd ) ) can probably fail due to different construction of loops,
                // so we check every edge of every loop below
                const auto len = calcPathLength( bd, *newMesh );
                const auto avgLen = len / bd.size();
                FillHoleNicelySettings settings
                {
                    .triangulateParams =
                    {
                        .metric = getUniversalMetric( *newMesh ),
                        .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong
                    },
                    .maxEdgeLen = 2 * (float)avgLen,
                    .edgeWeights = settings_.edgeWeights
                };
                settings.onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
                {
                    if ( newFaceSelection.test( newMesh->topology.left( e ) ) )
                        newFaceSelection.autoResizeSet( newMesh->topology.left( e1 ) );
                    if ( newFaceSelection.test( newMesh->topology.right( e ) ) )
                        newFaceSelection.autoResizeSet( newMesh->topology.right( e1 ) );
                    // if we split an edge with both unchangeable end vertices, then mark new vertex as unchangeable as well
                    if ( unchangeableVerts_.test( newMesh->topology.org( e1 ) ) &&
                         unchangeableVerts_.test( newMesh->topology.dest( e ) ) )
                        unchangeableVerts_.autoResizeSet( newMesh->topology.org( e ) );
                };
                for ( auto e : bd )
                    if ( !newMesh->topology.left( e ) )
                        fillHoleNicely( *newMesh, e, settings );
            }

            if ( newFaceSelection != obj_->getSelectedFaces() )
                AppendHistory<ChangeMeshFaceSelectionAction>( "Change Face Selection", obj_, std::move( newFaceSelection ) );
            if ( newEdgeSelection != obj_->getSelectedEdges() )
                AppendHistory<ChangeMeshEdgeSelectionAction>( "Change Edge Selection", obj_, std::move( newEdgeSelection ) );

            // newFaces include both faces inside the patch and subdivided faces around
            const FaceBitSet newFaces = newMesh->topology.getValidFaces() - oldFaces;
            auto meshAttribs = projectMeshAttributes( *obj_, MeshPart( *newMesh, &newFaces ) );

            appendMeshChangeHistory_( std::move( newMesh ), newFaces );

            if ( meshAttribs )
                emplaceMeshAttributes( obj_, std::move( *meshAttribs ) );

            reallocData_( obj_->mesh()->topology.lastValidVert() + 1 );
            sameValidVerticesAsInOriginMesh_ = originalMesh_->topology.getValidVerts() == obj_->mesh()->topology.getValidVerts();
            setDeviationCalculationMethod( deviationCalculationMethod_ );
            obj_->setDirtyFlags( DIRTY_ALL );

            // otherwise whole surface becomes red after patch and before mouse move
            updateRegion_( mousePos_ );
        }
    }
    else if ( ( settings_.workMode == WorkMode::Add || settings_.workMode == WorkMode::Remove ) &&
        settings_.relaxForceAfterEdit > 0.f && generalEditingRegion_.any() )
    {
        ownMeshChangedSignal_ = true;

        MeshRelaxParams params;
        params.region = &generalEditingRegion_;
        params.force = settings_.relaxForceAfterEdit;
        params.iterations = 5;
        relax( *obj_->varMesh(), params );
        updateValueChanges_( generalEditingRegion_ );
        obj_->setDirtyFlags( DIRTY_POSITION );
    }

    generalEditingRegion_.clear();
    generalEditingRegion_.resize( numV, false );

    removeLastStableObjMesh_();

    return true;
}

bool SurfaceManipulationWidget::onMouseMove_( int mouse_x, int mouse_y )
{
    auto mousePos = Vector2f{ float( mouse_x ), float( mouse_y ) };
    if ( !mousePressed_ )
        updateRegion_( mousePos );
    else
    {
        if ( settings_.workMode == WorkMode::Laplacian )
        {
            if ( appendHistoryAction_ )
            {
                appendHistoryAction_ = false;
                AppendHistory( historyAction_ );
            }
            laplacianMoveVert_( mousePos );
        }
        else
        {
            updateRegion_( mousePos );
            changeSurface_();
        }
    }

    return true;
}

void SurfaceManipulationWidget::postDraw_()
{
    if ( !badRegion_ )
        return;

    auto drawList = ImGui::GetBackgroundDrawList();
    const auto& mousePos = Vector2f( getViewerInstance().mouseController().getMousePos() );
    drawList->AddCircleFilled( ImVec2( mousePos.x, mousePos.y ), 10.f, Color::gray().getUInt32() );
}

void SurfaceManipulationWidget::appendMeshChangeHistory_( std::shared_ptr<Mesh> newMesh, const FaceBitSet& )
{
    AppendHistory( std::make_shared<PartialChangeMeshAction>( "mesh", obj_, setNew, std::move( newMesh ) ) );
}

void SurfaceManipulationWidget::reallocData_( size_t size )
{
    singleEditingRegion_.resize( size, false );
    visualizationRegion_.resize( size, false );
    generalEditingRegion_.resize( size, false );
    activePickedVertices_.resize( size, false );
    pointsShift_.resize( size, 0.f );
    editingDistanceMap_.resize( size, 0.f );
    visualizationDistanceMap_.resize( size, FLT_MAX );
    changedRegion_.resize( size, false );
    valueChanges_.resize( size, 0.f );
}

void SurfaceManipulationWidget::clearData_()
{
    singleEditingRegion_.clear();
    visualizationRegion_.clear();
    generalEditingRegion_.clear();
    pointsShift_.clear();
    editingDistanceMap_.clear();
    visualizationDistanceMap_.clear();
    changedRegion_.clear();
    valueChanges_.clear();
    activePickedVertices_.clear();
}

void SurfaceManipulationWidget::initConnections_()
{
    if ( connectionsInitialized_ )
        return;
    connectionsInitialized_ = true;
    meshChangedConnection_ = obj_->meshChangedSignal.connect( [&] ( uint32_t )
    {
        if ( ownMeshChangedSignal_ )
        {
            ownMeshChangedSignal_ = false;
            return;
        }
        abortEdit_();
        reallocData_( obj_->mesh()->topology.lastValidVert() + 1 );
        if ( settings_.workMode == WorkMode::Patch )
            updateUVmap_( false, true );
        sameValidVerticesAsInOriginMesh_ = originalMesh_->topology.getValidVerts() == obj_->mesh()->topology.getValidVerts();
        setDeviationCalculationMethod( deviationCalculationMethod_ );
        updateRegion_( Vector2f( getViewerInstance().mouseController().getMousePos() ) );
    } );
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceManipulationWidget::resetConnections_()
{
    connectionsInitialized_ = false;
    meshChangedConnection_.disconnect();
    disconnect();
}

void SurfaceManipulationWidget::changeSurface_()
{
    if ( !singleEditingRegion_.any() || badRegion_ )
        return;

    if ( appendHistoryAction_ )
    {
        appendHistoryAction_ = false;
        AppendHistory( historyAction_ );
    }

    MR_TIMER;

    if ( settings_.workMode == WorkMode::Patch )
    {
        generalEditingRegion_ |= singleEditingRegion_;
        return; // everything is done on mouse up
    }
    ownMeshChangedSignal_ = true;

    if ( settings_.workMode == WorkMode::Relax )
    {
        MeshRelaxParams params;
        params.region = &singleEditingRegion_;
        params.force = settings_.relaxForce;
        relax( *obj_->varMesh(), params );
        obj_->setDirtyFlags( DIRTY_POSITION );
        updateValueChanges_( singleEditingRegion_ );
        return;
    }

    Vector3f normal;
    auto objMeshPtr = lastStableObjMesh_ ? lastStableObjMesh_ : obj_;
    const auto& mesh = *objMeshPtr->mesh();
    for ( auto v : singleEditingRegion_ )
        normal += mesh.dirDblArea( v );
    normal = normal.normalized();

    auto& points = obj_->varMesh()->points;

    const float maxShift = settings_.editForce;
    const float intensity = ( 100.f - settings_.sharpness ) / 100.f * 0.5f + 0.25f;
    const float a1 = -1.f * ( 1 - intensity ) / intensity / intensity;
    const float a2 = intensity / ( 1 - intensity ) / ( 1 - intensity );
    const float direction = settings_.workMode == WorkMode::Remove ? -1.f : 1.f;
    BitSetParallelFor( singleEditingRegion_, [&] ( VertId v )
    {
        const float r = std::clamp( editingDistanceMap_[v] / settings_.radius, 0.f, 1.f );
        const float k = r < intensity ? a1 * r * r + 1 : a2 * ( r - 1 ) * ( r - 1 ); // I(r)
        float pointShift = maxShift * k; // shift = F * I(r)
        if ( pointShift > pointsShift_[v] )
        {
            pointShift -= pointsShift_[v];
            pointsShift_[v] += pointShift;
        }
        else
            return;
        points[v] += direction * pointShift * normal;
    } );
    generalEditingRegion_ |= singleEditingRegion_;
    changedRegion_ |= singleEditingRegion_;
    updateValueChanges_( singleEditingRegion_ );
    obj_->setDirtyFlags( DIRTY_POSITION );
}

void SurfaceManipulationWidget::updateUVmap_( bool set, bool wholeMesh )
{
    VertUVCoords uvs;
    obj_->updateAncillaryUVCoords( uvs );
    uvs.resizeWithReserve( obj_->mesh()->points.size(), UVCoord{ 0.5f, 1 } );
    const float normalize = 0.5f / settings_.radius;
    BitSetParallelFor( wholeMesh ? obj_->mesh()->topology.getValidVerts() : visualizationRegion_, [&] ( VertId v )
    {
        if ( set )
            uvs[v] = UVCoord( palette_->getUVcoord( valueChanges_[v], true ).x, ( visualizationDistanceMap_[v] * normalize - 0.5f ) * 100 + 0.5f );
        else
            uvs[v] = UVCoord( palette_->getUVcoord( valueChanges_[v], true ).x, 1.f );
    } );
    obj_->setAncillaryUVCoords( std::move( uvs ) );
}

void SurfaceManipulationWidget::updateRegion_( const Vector2f& mousePos )
{
    MR_TIMER;

    const auto& viewerRef = getViewerInstance();
    const ViewportId viewportId = viewerRef.viewport().id;
    std::vector<Vector2f> viewportPoints;
    if ( !mousePressed_ || ( mousePos - mousePos_ ).lengthSq() < 16.f )
        viewportPoints.push_back( Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewportId ) ) );
    else
    {
        // if the mouse shift is large, then the brush area is defined as the common area of many small shifts (creating many intermediate points of movement)
        const Vector2f newMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewportId ) );
        const Vector2f oldMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos_ ), viewportId ) );
        const Vector2f vec = newMousePos - oldMousePos;
        const int count = int( std::ceil( vec.length() ) ) + 1;
        const Vector2f step = vec / ( count - 1.f );
        viewportPoints.resize( count );
        for ( int i = 0; i < count; ++i )
            viewportPoints[i] = oldMousePos + step * float( i );
    }
    mousePos_ = mousePos;

    auto objMeshPtr = lastStableObjMesh_ ? lastStableObjMesh_ : obj_;
    std::vector<ObjAndPick> movedPosPick;
    if ( ignoreOcclusion_ )
        movedPosPick = getViewerInstance().viewport().multiPickObjects( { { static_cast< VisualObject* >( objMeshPtr.get() ) } }, viewportPoints );
    else
    {
        const auto visualObjectsS = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selectable>();
        std::vector<VisualObject*> visualObjectsP;
        visualObjectsP.reserve( visualObjectsS.size() );
        for ( int i = 0; i < visualObjectsS.size(); ++i )
        {
            if ( lastStableObjMesh_ && visualObjectsS[i] == obj_ )
                visualObjectsP.push_back( lastStableObjMesh_.get() );
            else if ( visualObjectsS[i]->isVisible( viewportId ) )
                visualObjectsP.push_back( visualObjectsS[i].get() );
        }
        movedPosPick = getViewerInstance().viewport().multiPickObjects( visualObjectsP, viewportPoints );
    }

    const auto& mesh = *objMeshPtr->mesh();
    activePickedVertices_.reset();
    for ( const auto& [obj,pick] : movedPosPick )
    {
        if ( !obj || obj != objMeshPtr )
            continue;
        activePickedVertices_.set( mesh.getClosestVertex( pick ) );
    }

    updateVizualizeSelection_();
    if ( !mousePressed_ )
    {
        editingDistanceMap_ = visualizationDistanceMap_;
        singleEditingRegion_ = visualizationRegion_;
    }
    else
    {
        std::vector<MeshTriPoint> triPoints;
        VertBitSet newVerts( singleEditingRegion_.size() );
        triPoints.reserve( movedPosPick.size() );
        for ( int i = 0; i < movedPosPick.size(); ++i )
        {
            if ( movedPosPick[i].first == objMeshPtr )
            {
                const auto& pick = movedPosPick[i].second;
                VertId v[3];
                mesh.topology.getTriVerts( pick.face, v );
                for ( int j = 0; j < 3; ++j )
                    newVerts.set( v[j] );
                triPoints.push_back( mesh.toTriPoint( pick.face, pick.point ) );
            }
        }

        if ( triPoints.size() == 1 )
        {
            // if the mouse shift is small (one point of movement), then the distance map of the points is calculated in 3d space (as visual more circular area)
            bool keepOld = settings_.workMode == WorkMode::Patch;
            activePickedVertices_.reset();
            activePickedVertices_.set( mesh.getClosestVertex( triPoints[0] ) );
            updateDistancesAndRegion_( mesh, activePickedVertices_, editingDistanceMap_, singleEditingRegion_, keepOld ? &generalEditingRegion_ : nullptr );
        }
        else
        {
            // if the mouse shift is large (more then one point of movement), then the distance map is calculated from the surface of the mesh
            // TODO try to rework with SpaceDistance (for multiple point. does not exist, need to create) if it's not slower
            singleEditingRegion_ = newVerts;
            dilateRegion( mesh, singleEditingRegion_, settings_.radius * 1.5f );
            editingDistanceMap_ = computeSurfaceDistances( mesh, triPoints, settings_.radius * 1.5f, &singleEditingRegion_ );
        }
    }
    for ( auto v : singleEditingRegion_ )
        singleEditingRegion_.set( v, editingDistanceMap_[v] <= settings_.radius );
    singleEditingRegion_ -= unchangeableVerts_;
}

void SurfaceManipulationWidget::abortEdit_()
{
    if ( !mousePressed_ )
        return;
    mousePressed_ = false;
    removeLastStableObjMesh_();
    appendHistoryAction_ = false;
    historyAction_.reset();
    generalEditingRegion_.clear();
}

void SurfaceManipulationWidget::laplacianPickVert_( const PointOnFace& pick )
{
    appendHistoryAction_ = true;
    storedDown_ = getViewerInstance().mouseController().getMousePos();
    const auto& mesh = *obj_->mesh();
    touchVertId_ = mesh.getClosestVertex( pick );
    touchVertIniPos_ = mesh.points[touchVertId_];
    laplacian_ = std::make_unique<Laplacian>( *obj_->varMesh() );
    laplacian_->init( singleEditingRegion_, settings_.edgeWeights );
    historyAction_ = std::make_shared<SmartChangeMeshPointsAction>( "Brush: Deform", obj_ );
    changedRegion_ |= singleEditingRegion_;
    createLastStableObjMesh_();
    lastStableValueChanges_ = valueChanges_;
}

void SurfaceManipulationWidget::laplacianMoveVert_( const Vector2f& mousePos )
{
    ownMeshChangedSignal_ = true;
    auto& viewerRef = getViewerInstance();
    const float zpos = viewerRef.viewport().projectToViewportSpace( obj_->worldXf()( touchVertIniPos_ ) ).z;
    auto viewportPoint1 = viewerRef.screenToViewport( Vector3f( mousePos.x, mousePos.y, zpos ), viewerRef.viewport().id );
    auto pos1 = viewerRef.viewport().unprojectFromViewportSpace( viewportPoint1 );
    auto viewportPoint0 = viewerRef.screenToViewport( Vector3f( float( storedDown_.x ), float( storedDown_.y ), zpos ), viewerRef.viewport().id );
    auto pos0 = viewerRef.viewport().unprojectFromViewportSpace( viewportPoint0 );
    const Vector3f move = obj_->worldXf().A.inverse()* ( pos1 - pos0 );
    laplacian_->fixVertex( touchVertId_, touchVertIniPos_ + move );
    laplacian_->apply();
    obj_->setDirtyFlags( DIRTY_POSITION );
    updateValueChanges_( singleEditingRegion_ );
}

void SurfaceManipulationWidget::updateVizualizeSelection_()
{
    bool keepOld = settings_.workMode == WorkMode::Patch && mousePressed_;
    if ( keepOld )
        visualizationRegion_ -= generalEditingRegion_;
    updateUVmap_( false );
    visualizationRegion_.reset();
    auto objMeshPtr = lastStableObjMesh_ ? lastStableObjMesh_ : obj_;
    const auto& mesh = *objMeshPtr->mesh();
    badRegion_ = false;
    if ( activePickedVertices_.none() )
        return;
    if ( settings_.workMode == WorkMode::Laplacian && unchangeableVerts_.intersects( activePickedVertices_ ) )
    {
        badRegion_ = true;
        return;
    }
    updateDistancesAndRegion_( mesh, activePickedVertices_, visualizationDistanceMap_, visualizationRegion_, keepOld ? &generalEditingRegion_ : nullptr );
    expand( mesh.topology, visualizationRegion_ );
    {
        int pointsCount = 0;
        for ( auto vId : visualizationRegion_ )
        {
            if ( visualizationDistanceMap_[vId] <= settings_.radius )
                ++pointsCount;
            if ( pointsCount == 3 )
                break;
        }
        badRegion_ = pointsCount < 3;
    }
    if ( !badRegion_ )
        updateUVmap_( true );
}

void SurfaceManipulationWidget::updateRegionUVs_( const VertBitSet& region )
{
    VertUVCoords uvs;
    obj_->updateAncillaryUVCoords( uvs );
    uvs.resizeWithReserve( obj_->mesh()->points.size(), UVCoord{ 0.5f, 1 } );
    BitSetParallelFor( region, [&] ( VertId v )
    {
        uvs[v].x = palette_->getUVcoord( valueChanges_[v], true ).x;
    } );
    obj_->setAncillaryUVCoords( std::move( uvs ) );
}

void SurfaceManipulationWidget::updateValueChanges_( const VertBitSet& region )
{
    switch ( deviationCalculationMethod_ )
    {
    case DeviationCalculationMethod::PointToPoint:
        updateValueChangesPointToPoint_( region );
        break;
    case DeviationCalculationMethod::PointToPlane:
        updateValueChangesPointToPlane_( region );
        break;
    case DeviationCalculationMethod::ExactDistance:
    default:
        updateValueChangesExactDistance_( region );
    }
}

void SurfaceManipulationWidget::updateValueChangesPointToPoint_( const VertBitSet& region )
{
    const auto& oldPoints = originalMesh_->points;
    const auto& mesh = *obj_->mesh();
    const auto& points = mesh.points;
    BitSetParallelFor( region, [&] ( VertId v )
    {
        const Vector3f shift = points[v] - oldPoints[v];
        const float sign = dot( shift, mesh.normal( v ) ) >= 0.f ? 1.f : -1.f;
        valueChanges_[v] = shift.length() * sign;
    } );

    updateRegionUVs_( region );
}

void SurfaceManipulationWidget::updateValueChangesPointToPlane_( const VertBitSet& region )
{
    const auto& oldMesh = *originalMesh_;
    const auto& oldPoints = oldMesh.points;
    const auto& mesh = *obj_->mesh();
    const auto& points = mesh.points;
    BitSetParallelFor( region, [&] ( VertId v )
    {
        const Plane3f plane = Plane3f::fromDirAndPt( oldMesh.normal( v ), oldPoints[v] );
        const float shift = plane.distance( points[v] );
        valueChanges_[v] = shift;
    } );

    updateRegionUVs_( region );
}

void SurfaceManipulationWidget::updateValueChangesExactDistance_( const VertBitSet& region )
{
    const auto& mesh = *obj_->mesh();
    const auto& meshVerts = mesh.points;

    std::vector<MeshProjectionResult> projResults( meshVerts.size() );
    BitSetParallelFor( region, [&] ( VertId v )
    {
        projResults[v] = findProjection( meshVerts[v], *originalMesh_, FLT_MAX, nullptr, 0 );
    } );

    unknownSign_.clear();
    unknownSign_.resize( meshVerts.size(), false );

    BitSetParallelFor( region, [&] ( VertId v )
    {
        const auto& projRes = projResults[v];
        auto res = projRes.distSq;
        if ( projRes.mtp.e )
            res = originalMesh_->signedDistance( meshVerts[VertId( v )], projRes );
        else
            res = std::sqrt( res );

        valueChanges_[v] = res;
        if ( !projRes.mtp )
            unknownSign_.set( v, true );
    } );

    BitSetParallelFor( unknownSign_, [&] ( VertId v )
    {
        float sumNeis = 0;
        for ( EdgeId e : orgRing( mesh.topology, v ) )
        {
            auto d = mesh.topology.dest( e );
            if ( !unknownSign_.test( d ) )
                sumNeis += valueChanges_[d];
        }
        if ( sumNeis < 0 )
            valueChanges_[v] = -valueChanges_[v];
    } );

    updateRegionUVs_( region );
}

}
