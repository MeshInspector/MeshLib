#include "MRScannerPlugin.h"
#include "MRScannerSchemaUi.h"
#include "MRBestView.h"
#include "MRICPDialog.h"
#include "MRStatePlugin.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "MRFileDialog.h"
#include "MRAppendHistory.h"
#include "ImGuiHelpers.h"
#include "MRUIStyle.h"
#include "MRRibbonConstants.h"
#include "MRCommandLoop.h"
#include "MRPlaneWidget.h"
#include "MRImGuiImage.h"
#include "MRRibbonButtonDrawer.h"
#include "MRRayWidget.h"
#include "MRI18n.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshOrPoints.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRChangeMeshAction.h"
#include "MRMesh/MRChangePointCloudAction.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRICP.h"
#include "MRMesh/MRGlobalRegistration.h"
#include "MRPch/MRSpdlog.h"
#include <imgui_internal.h>

#include <algorithm>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>

namespace MR
{

namespace
{

// Clip ObjectPoints or ObjectMesh with a plane, removing one part
void clipScan( std::shared_ptr<VisualObject> scan, Plane3f plane, bool history )
{
    plane = invTransformed( plane, scan->worldXf() );
    if ( std::shared_ptr<ObjectPoints> objPoints = std::dynamic_pointer_cast<ObjectPoints>( scan ) )
    {
        auto h = !history ? nullptr :
            std::make_unique<Historian<ChangePointCloudAction>>( _t( "Clip Scan by Plane" ), objPoints );

        PointCloud& cloud = *objPoints->varPointCloud();
        VertBitSet validPoints = cloud.validPoints;
        BitSetParallelFor( cloud.validPoints, [&]( VertId v )
            {
                if ( plane.distance( cloud.points[v] ) < 0 )
                    validPoints.reset( v );
            } );
        cloud.validPoints = validPoints;
        cloud.pack();
    }
    else if ( std::shared_ptr<ObjectMesh> objMesh = std::dynamic_pointer_cast<ObjectMesh>( scan ) )
    {
        auto h = !history ? nullptr :
            std::make_unique<Historian<ChangeMeshAction>>( _t( "Clip Scan by Plane" ), objMesh );

        // Simple clipping, just remove whole faces
        Mesh& mesh = *objMesh->varMesh();
        VertBitSet invalidPoints;
        invalidPoints.resize( mesh.topology.getValidVerts().size(), false );
        BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
            {
                if ( plane.distance( mesh.points[v] ) < 0 )
                    invalidPoints.set( v );
            } );
        mesh.topology.deleteFaces( getIncidentFaces( mesh.topology, invalidPoints ) );
        mesh.pack();
    }
}

// Helper class to accumulate mesh parts or points
class MeshAndPoints
{
    Mesh mesh_;
    PointCloud pointCloud_;

    void add( const Mesh& mesh, const AffineXf3f& xf )
    {
        VertBitSet newVerts( mesh_.topology.vertSize() );
        mesh_.addMesh( mesh ); // appends only valid vertices
        newVerts.resize( mesh_.topology.vertSize(), true );
        if ( xf != AffineXf3f() )
            mesh_.transform( xf, &newVerts );
    }

    void add( const PointCloud& points, const AffineXf3f& xf )
    {
        const bool addNormals = pointCloud_.hasNormals() && points.hasNormals();
        Matrix3f normMat;
        if ( addNormals )
            normMat = xf.A.inverse().transposed();

        pointCloud_.points.reserve( pointCloud_.points.size() + points.points.size() );
        for ( auto v : points.validPoints )
        {
            pointCloud_.points.push_back( xf( points.points[v] ) );
            if ( addNormals )
                pointCloud_.normals.push_back( normMat * points.normals[v] );
        }
        pointCloud_.validPoints.resize( pointCloud_.points.size(), true );
    }

public:
    bool hasMesh()
    {
        return mesh_.topology.numValidVerts() != 0;
    }
    bool hasPoints()
    {
        return !pointCloud_.points.empty();
    }
    bool isEmpty()
    {
        return !hasMesh() && !hasPoints();
    }
    MeshOrPoints meshOrPoints()
    {
        // Should not have both
        return hasMesh() ? MeshOrPoints( mesh_ ) : MeshOrPoints( pointCloud_ );
    }

    void add( const VisualObject* obj )
    {
        if ( auto objMesh = dynamic_cast<const ObjectMesh*>( obj ) )
            add( *objMesh->meshPtr(), obj->worldXf() );
        else if ( auto objPnts = dynamic_cast<const ObjectPoints*>( obj ) )
            add( *objPnts->pointCloudPtr(), obj->worldXf() );
    }
};
// The host's own scan processing settings. They are stored in the backend's Config object next to the
// backend's settings.
struct ProcessingSettings
{
    // Rotate scans after each take except the one taken (e.g. to use with a rotary table)
    bool rotate = false;
    int rotateDegrees = 15;
    Vector3f rotateOrg;
    bool rotateOrgSet = false;
    Vector3f rotateAxis = Vector3f::plusZ();

    // After each capture, register the scan to the previous scans using ICP
    bool registration = false;

    void load( const std::string& configKey )
    {
        Json::Value json = Config::instance().getJsonValue( configKey );
        if ( json["rotate"] )
            rotate = json["rotate"].asBool();
        if ( json["rotateDegrees"] )
            rotateDegrees = json["rotateDegrees"].asInt();
        if ( json["rotateOrg"] )
            deserializeFromJson( json["rotateOrg"], rotateOrg );
        if ( json["rotateOrgSet"] )
            rotateOrgSet = json["rotateOrgSet"].asBool();
        if ( json["rotateAxis"] )
            deserializeFromJson( json["rotateAxis"], rotateAxis );
        if ( json["registration"] )
            registration = json["registration"].asBool();
    }

    // rotationOnly keeps the saved registration flag (the Rot/Reg tab's Save)
    void save( const std::string& configKey, bool rotationOnly = false ) const
    {
        Config& config = Config::instance();
        Json::Value json = config.getJsonValue( configKey );
        json["rotate"] = rotate;
        json["rotateDegrees"] = rotateDegrees;
        serializeToJson( rotateOrg, json["rotateOrg"] );
        json["rotateOrgSet"] = rotateOrgSet;
        serializeToJson( rotateAxis, json["rotateAxis"] );
        if ( !rotationOnly )
            json["registration"] = registration;
        config.setJsonValue( configKey, json );
    }

    // Change rotateDegrees to the next/previous value in a predefined list of angles
    static constexpr int cRotateDegrees[] = { -90, -75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90 };
    void nextRotateDegrees()
    {
        for ( int v : cRotateDegrees )
            if ( v > rotateDegrees )
            {
                rotateDegrees = v;
                break;
            }
    }
    void prevRotateDegrees()
    {
        int vp = cRotateDegrees[0];
        for ( int v : cRotateDegrees )
            if ( v >= rotateDegrees )
            {
                rotateDegrees = vp;
                break;
            }
            else
                vp = v;
    }
};

// The Config key of the selected backend's name
constexpr const char* cBackendKey = "scannerBackend";

// Generic scanner plugin: lists every backend, communicates with the selected one only through
// IScannerBackend, renders the backend's parameter schema generically, and hosts the backend-agnostic
// scan processing (clip / rotate / register gizmos, scene insertion) plus a preview of the backend's
// 2D images and a capability-gated live region.
class ScannerPlugin : public StatePlugin
{
public:
    explicit ScannerPlugin( std::vector<std::unique_ptr<IScannerBackend>> backends );
    virtual void drawDialog( ImGuiContext* ) override;

private:
    IScannerBackend* backend_ = nullptr; // the selected one of backends_; only its runtime is loaded
    bool loadCancelled_ = false; // Shift was held when the tool opened, so the runtime was not loaded
    std::string error_;          // why the last operation failed, empty after one succeeded
    ProcessingSettings settings_; // persist in the backend's Config object
    std::string configKey_() const { return toLower( backend_->name() ); }

    // ---- preview / live ----
    std::deque<ImGuiImage> images_; // only grows: a texture must outlive the frame that draws it
    size_t numImages_ = 0;          // the current image set is the first numImages_ of images_
    bool imagesDirty_{};
    bool isWindowOpen_ = false;
    bool wasLive_ = false;
    ImVec2 windowSize_{};
    std::mutex streamMutex_;
    bool frameQueued_ = false; // guarded by streamMutex_: a GUI command will take the latest frame
    std::vector<Image> streamImages_;
    PointCloud streamPoints_;
    std::shared_ptr<ObjectPoints> streamObject_;
    // destroyed before the members above: a backend stopping its stream may deliver one more frame
    std::vector<std::unique_ptr<IScannerBackend>> backends_;

    // ---- tabs ----
    bool selectFirstTab_ = true; // open on Parameters
    bool scanTabActive_ = false; // the preview is on the Current Scan tab; elsewhere it pops out
    float paramsEndY_ = 0;       // the Parameters page height: the capture block never sits higher

    // ---- scan processing (backend-agnostic) ----
    PlaneWidget planeWidget_;
    bool planeWidgetActive_{};
    RayWidget axisWidget_;
    std::vector<std::weak_ptr<VisualObject>> scans_;
    bool firstScan_ = true;
    int scansCounter_ = 0;
    struct
    {
        ICPProperties params = { .method = ICPMethod::Combined, .farDistFactor = 1.5f, .iterLimit = 100, .badIterStopCount = 20 };
        float samplingVoxelSize = 0.0f;
        float objectSize = 0.0f;
        bool ransac = false; // also try the RANSAC initial position by shape descriptors, then ICP
        // the modes that accept the rigid pose RANSAC finds; the others constrain each step from the current pose
        bool rigidMode() const { return params.icpMode == ICPMode::RigidScale || params.icpMode == ICPMode::AnyRigidXf; }
        std::string result;
        struct ICPInfo { size_t numSamples = 0, numPairs = 0; float meanSqDistToPoint = 0, meanSqDistToPlane = 0; };
        std::optional<ICPInfo> info;
    } icp_;

    virtual bool onEnable_() override;
    virtual bool onDisable_() override;

    void openBackend_( bool skipLoad ); // loads the runtime of backend_ unless skipLoad
    void closeBackend_();               // unloads it and drops its images and stream
    void selectBackend_( IScannerBackend& backend );
    void loadRuntime_( const std::string& dir ); // then connect
    void onScanProduced_( std::shared_ptr<VisualObject> obj );
    void onLiveFrame_( ScannerLiveFrame&& frame );
    void clearStream_(); // after the stream stopped: drop the undelivered frame and the live cloud
    std::vector<std::shared_ptr<VisualObject>> validScans_() const;
    bool getScansCenterPoint_( Vector3f& c );
    void updateScansBox_();
    void rotateScans_();
    void registerScan_( std::shared_ptr<VisualObject> scan );
    void orientCameraToObject_( VisualObject& obj );
    void setImage_( size_t i, Image img );
    void refreshPreview_();
    bool hasImage_() const; // the preview shows only when an image has pixels
    ImVec2 imageSize_( float rowWidth ) const;
    void drawImages_( ImVec2 size );
    void drawBackendCombo_();
    void drawNotLoaded_();
    void drawClipPlane_();
    void drawRotReg_();
    void drawPreview_();
    void drawPreviewWindow_();
};

ScannerPlugin::ScannerPlugin( std::vector<std::unique_ptr<IScannerBackend>> backends ) :
    StatePlugin( "Scanner", StatePluginTabs::Other ),
    backends_( std::move( backends ) )
{
    backend_ = backends_.front().get(); // makeScannerPlugin passes at least one
    for ( auto& backend : backends_ )
        backend->setLiveFrameCallback( [this]( ScannerLiveFrame&& f ) { onLiveFrame_( std::move( f ) ); } );
    CommandLoop::appendCommand( [this]()
    {
        std::string selected = Config::instance().getJsonValue( cBackendKey ).asString();
        for ( auto& backend : backends_ )
        {
            backend->loadSettings();
            // none chosen yet (before this combo): the first backend with its own Config key, e.g. its SDK path
            if ( selected.empty() && Config::instance().hasJsonValue( toLower( backend->name() ) ) )
                selected = backend->name();
            if ( backend->name() == selected )
                backend_ = backend.get();
        }
        settings_.load( configKey_() );
    } );
}

bool ScannerPlugin::onEnable_()
{
    // hold Shift while opening the tool to skip loading the SDK, e.g. one that crashes the application
    openBackend_( ImGui::IsKeyDown( ImGuiMod_Shift ) );

    scansCounter_ = 0;
    updateScansBox_();

    axisWidget_.create( { settings_.rotateOrg, settings_.rotateAxis }, 100.0f,
        [this]( const Line3f& ray, bool )
        {
            settings_.rotateOrg = ray.p;
            settings_.rotateAxis = ray.d;
            if ( ray.p != Vector3f{} )
                settings_.rotateOrgSet = true;
        } );
    return true;
}

bool ScannerPlugin::onDisable_()
{
    for ( std::shared_ptr<VisualObject> scan : validScans_() )
        scan->select( true );
    axisWidget_.reset();
    planeWidget_.disconnect();
    planeWidget_.setOnPlaneUpdateCallback( nullptr );
    planeWidget_.undefinePlane();
    planeWidgetActive_ = false;
    scans_.clear();
    closeBackend_();
    icp_ = {};
    return true;
}

void ScannerPlugin::openBackend_( bool skipLoad )
{
    loadCancelled_ = skipLoad;
    if ( !loadCancelled_ )
        loadRuntime_( backend_->runtimeDir() );
    firstScan_ = true; // its first scan sets the pivot unless saved, and fits the view
    imagesDirty_ = true;
    selectFirstTab_ = true;
    paramsEndY_ = 0;
    wasLive_ = false;
}

void ScannerPlugin::closeBackend_()
{
    backend_->saveSettings(); // all of them, to open with them next time
    backend_->unloadRuntime(); // stops the stream first, so clearStream_ drops its last frame
    for ( ImGuiImage& image : images_ )
        image.update( {} );
    clearStream_();
    isWindowOpen_ = false;
}

void ScannerPlugin::selectBackend_( IScannerBackend& backend )
{
    closeBackend_();
    backend_ = &backend;
    Config::instance().setJsonValue( cBackendKey, backend.name() );
    settings_ = {}; // load() keeps the fields missing in the Config
    settings_.load( configKey_() );
    axisWidget_.updateRay( { settings_.rotateOrg, settings_.rotateAxis } );
    openBackend_( false );
}

void ScannerPlugin::loadRuntime_( const std::string& dir )
{
    // a missing or busy scanner leaves the runtime loaded: its failure shows under the capture block
    Expected<void> res = backend_->loadRuntime( dir );
    if ( res )
        res = backend_->connect();
    error_ = res ? std::string() : std::move( res.error() );
}

std::vector<std::shared_ptr<VisualObject>> ScannerPlugin::validScans_() const
{
    std::vector<std::shared_ptr<VisualObject>> scans;
    scans.reserve( scans_.size() );
    for ( const std::weak_ptr<VisualObject>& weakScan : scans_ )
        if ( std::shared_ptr<VisualObject> scan = weakScan.lock();
            scan && scan->isAncestor( &SceneRoot::get() ) )
            scans.push_back( scan );
    return scans;
}

bool ScannerPlugin::getScansCenterPoint_( Vector3f& o )
{
    Box3f box;
    for ( std::shared_ptr<VisualObject> scan : validScans_() )
        box.include( scan->getWorldBox() );
    if ( box.valid() )
        o = box.center();
    return box.valid();
}

void ScannerPlugin::updateScansBox_()
{
    Box3f box;
    for ( std::shared_ptr<VisualObject> scan : validScans_() )
        box.include( scan->getWorldBox() );
    if ( streamObject_ )
        box.include( streamObject_->getWorldBox() );
    planeWidget_.updateBox( box.valid() ? box :
        Box3f( Vector3f::diagonal( -100.0f ), Vector3f::diagonal( 100.0f ) ) );
}

void ScannerPlugin::rotateScans_()
{
    AffineXf3f xf = AffineXf3f::linear( Matrix3f::rotation( settings_.rotateAxis,
        -settings_.rotateDegrees * ( PI_F / 180.0f ) ) );
    xf = AffineXf3f::translation( settings_.rotateOrg ) * xf * AffineXf3f::translation( -settings_.rotateOrg );
    for ( std::shared_ptr<VisualObject> scan : validScans_() )
    {
        AppendHistory<ChangeXfAction>( _t( "Rotate scan" ), scan );
        scan->setXf( xf * scan->xf() );
    }
}

void ScannerPlugin::registerScan_( std::shared_ptr<VisualObject> scan )
{
    std::shared_ptr<ObjectMesh> scanMeshObj = std::dynamic_pointer_cast<ObjectMesh>( scan );
    std::shared_ptr<ObjectPoints> scanPointsObj = std::dynamic_pointer_cast<ObjectPoints>( scan );
    MeshAndPoints fixedMerged;

    const bool wasEmpty = icp_.objectSize == 0.0f;
    for ( std::shared_ptr<VisualObject> fixed : validScans_() )
    {
        if ( bool( fixed->asType<ObjectPoints>() ) == bool( scanPointsObj ) )
            fixedMerged.add( fixed.get() );
        icp_.objectSize = std::max( icp_.objectSize, fixed->getWorldBox().diagonal() );
    }
    if ( wasEmpty )
    {
        icp_.params.distThresholdSq = roundToPrecision( sqr( icp_.objectSize ), 3 );
        icp_.samplingVoxelSize = roundToPrecision( icp_.objectSize * 0.015f, 3 );
    }

    if ( fixedMerged.isEmpty() || icp_.samplingVoxelSize == 0.0f || ( !scanPointsObj && !scanMeshObj ) )
        return;
    const MeshOrPoints scanMp = scanPointsObj ? MeshOrPoints( *scanPointsObj->pointCloudPtr() ) : MeshOrPoints( *scanMeshObj->meshPtr() );
    const MeshOrPoints fixedMp = fixedMerged.meshOrPoints();
    ICP icp( scanMp, fixedMp, scan->worldXf(), AffineXf3f(), icp_.samplingVoxelSize );
    icp.setParams( icp_.params );
    AffineXf3f xf = icp.calculateTransformation();
    icp_.result = icp.getStatusInfo();
    if ( icp_.ransac && icp_.rigidMode() )
    {
        // also from any position; the merged previous scans are object 0, which keeps its place
        Vector<MeshOrPointsXf, ObjId> objs;
        objs.push_back( { fixedMp, {} } );
        objs.push_back( { scanMp, scan->worldXf() } );
        const auto res = globalRegistration( objs, { .samplingVoxelSize = icp_.samplingVoxelSize } );
        if ( !res )
            icp_.result += "\nRANSAC: " + res.error();
        else if ( res->groupRoots[ObjId( 1 )] != ObjId( 0 ) )
            icp_.result += "\nRANSAC: no overlap with the previous scans found";
        else
        {
            // the ICP pose refined with the threshold globalRegistration refines its pose with,
            // then the pose with clearly more samples within one voxel of the other surface wins
            ICPProperties params = icp_.params;
            params.distThresholdSq = sqr( 5 * icp_.samplingVoxelSize );
            icp.setParams( params );
            icp.setFloatXf( xf );
            xf = icp.calculateTransformation();
            params.distThresholdSq = sqr( icp_.samplingVoxelSize );
            params.farDistFactor = std::numeric_limits<float>::max();
            icp.setParams( params );
            auto numClose = [&] ( const AffineXf3f& fltXf )
            {
                icp.setFloatXf( fltXf );
                icp.updatePointPairs();
                return icp.getNumActivePairs();
            };
            const size_t icpClose = numClose( xf ), ransacClose = numClose( res->xfs[ObjId( 1 )] );
            const bool applied = ransacClose > icpClose + icpClose / 4; // a tie or a small gain keeps the ICP pose
            if ( applied )
                xf = res->xfs[ObjId( 1 )];
            icp_.result += fmt::format( "\nRANSAC initial position {}:\n{} close pairs vs {} by ICP alone",
                applied ? "applied" : "discarded", ransacClose, icpClose );
            icp.setParams( icp_.params );
            icp.setFloatXf( xf );
            icp.updatePointPairs();
        }
    }
    icp_.info = { .numSamples = icp.getNumSamples(), .numPairs = icp.getNumActivePairs(),
        .meanSqDistToPoint = icp.getMeanSqDistToPoint(), .meanSqDistToPlane = icp.getMeanSqDistToPlane() };

    AppendHistory<ChangeXfAction>( _t( "Register scan" ), scan );
    scan->setXf( xf ); // the new to-world pose, as the scan has no parent yet
}

void ScannerPlugin::orientCameraToObject_( VisualObject& obj )
{
    const Vector3f surfaceVector = obj.xf().A.inverse().transposed() * findAverageSurfaceVector( obj );
    const Vector3f up = backend_->upDirection(), bestDir = surfaceVector;
    if ( surfaceVector.lengthSq() < 0.1f || std::abs( dot( surfaceVector, up ) ) > 0.9f )
        return;
    getViewerInstance().viewport().cameraLookAlong( -bestDir, cross( bestDir, cross( up, bestDir ) ) );
}

void ScannerPlugin::onScanProduced_( std::shared_ptr<VisualObject> obj )
{
    if ( !obj )
        return;
    obj->setName( "Scan" + std::to_string( scansCounter_++ ) );

    for ( auto& o : getAllObjectsInTree<VisualObject>( &SceneRoot::get() ) )
        o->select( false );
    obj->select( true );
    scans_.push_back( obj );

    if ( planeWidget_.getPlaneObject() )
        clipScan( obj, planeWidget_.getPlane(), false );

    SCOPED_HISTORY( _t( "Import scan" ) );
    if ( settings_.rotate && settings_.rotateDegrees != 0 )
        rotateScans_();
    if ( settings_.registration )
        registerScan_( obj );

    AppendHistory<ChangeSceneAction>( "Import scan", obj, ChangeSceneAction::Type::AddObject );
    SceneRoot::get().addChild( obj );

    orientCameraToObject_( *obj.get() );
    if ( firstScan_ )
    {
        getViewerInstance().preciseFitDataViewport();
        if ( Box3f box = obj->getBoundingBox(); box.valid() )
        {
            axisWidget_.updateLength( box.diagonal() * 0.8f );
            if ( !settings_.rotateOrgSet )
                axisWidget_.updateOrigin( settings_.rotateOrg = box.center() );
            settings_.rotateOrgSet = true;
        }
        firstScan_ = false;
    }
}

void ScannerPlugin::onLiveFrame_( ScannerLiveFrame&& frame )
{
    {
        std::lock_guard<std::mutex> lock( streamMutex_ );
        streamImages_.resize( std::max( streamImages_.size(), frame.images.size() ) );
        for ( size_t i = 0; i < frame.images.size(); i++ )
            if ( !frame.images[i].pixels.empty() ) // an empty one keeps a pending image
                streamImages_[i] = std::move( frame.images[i] );
        if ( frame.points )
            streamPoints_ = std::move( *frame.points );
        if ( frameQueued_ )
            return;
        frameQueued_ = true;
    }
    // queue, don't block: stopLivePreview() on the GUI thread waits for this SDK thread to return,
    // so waiting here for the GUI thread would deadlock
    CommandLoop::appendCommand( [this]()
    {
        std::lock_guard<std::mutex> lock( streamMutex_ );
        frameQueued_ = false;
        bool delivered = false;
        for ( size_t i = 0; i < streamImages_.size(); i++ )
            if ( !streamImages_[i].pixels.empty() )
            {
                setImage_( i, std::move( streamImages_[i] ) );
                streamImages_[i] = {};
                delivered = true;
            }
        if ( !delivered && streamPoints_.points.empty() )
            return; // cleared by clearStream_ after the stream stopped
        imagesDirty_ = false;
        if ( !scanTabActive_ && !backend_->livePreviewActive() )
            isWindowOpen_ = true; // a single captured frame; a stream pops out once, in drawDialog
        getViewerInstance().postEmptyEvent();
        if ( streamPoints_.points.size() && backend_->livePreviewActive() )
        {
            if ( !streamObject_ )
            {
                streamObject_ = std::make_shared<ObjectPoints>();
                streamObject_->setAncillary( true );
                SceneRoot::get().addChild( streamObject_ );
            }
            streamObject_->setPointCloud( std::make_shared<PointCloud>( std::move( streamPoints_ ) ) );
            if ( planeWidget_.getPlaneObject() )
                clipScan( streamObject_, planeWidget_.getPlane(), false );
            streamPoints_ = {};
            if ( firstScan_ )
                getViewerInstance().preciseFitDataViewport();
            firstScan_ = false;
        }
    } );
}

void ScannerPlugin::clearStream_()
{
    {
        std::lock_guard<std::mutex> lock( streamMutex_ );
        for ( auto& img : streamImages_ )
            img = {};
        streamPoints_ = {};
    }
    if ( streamObject_ )
    {
        streamObject_->detachFromParent();
        streamObject_.reset();
    }
    updateScansBox_();
}

void ScannerPlugin::setImage_( size_t i, Image img )
{
    while ( images_.size() <= i )
        images_.emplace_back();
    images_[i].update( { std::move( img ) } );
    numImages_ = std::max( numImages_, i + 1 );
}

void ScannerPlugin::refreshPreview_()
{
    std::vector<Image> imgs = backend_->previewImages();
    numImages_ = 0;
    for ( size_t i = 0; i < imgs.size(); i++ )
        setImage_( i, std::move( imgs[i] ) );
    imagesDirty_ = false;
}

bool ScannerPlugin::hasImage_() const
{
    return std::any_of( images_.begin(), images_.begin() + std::ptrdiff_t( numImages_ ),
        []( const ImGuiImage& image ) { return image.getImageWidth() > 0; } );
}

ImVec2 ScannerPlugin::imageSize_( float rowWidth ) const
{
    const float n = float( numImages_ );
    const float w = floor( ( rowWidth - ImGui::GetStyle().ItemSpacing.x * ( n - 1 ) ) / n );
    const ImGuiImage& first = images_.front(); // sets the aspect of all
    return { w, first.getImageWidth() > 0 ? floor( w * first.getImageHeight() / first.getImageWidth() ) : w };
}

void ScannerPlugin::drawImages_( ImVec2 size )
{
    for ( size_t i = 0; i < numImages_; i++ )
    {
        if ( i > 0 )
            ImGui::SameLine();
        ImGui::Image( images_[i], size );
    }
}

void ScannerPlugin::drawBackendCombo_()
{
    std::vector<std::string> names;
    int selected = 0;
    for ( const auto& backend : backends_ )
    {
        if ( backend.get() == backend_ )
            selected = int( names.size() );
        names.push_back( backend->name() );
    }
    if ( UI::combo( _tr( "Scanner" ), &selected, names ) && backends_[selected].get() != backend_ )
        selectBackend_( *backends_[selected] );
}

void ScannerPlugin::drawNotLoaded_()
{
    const ImGuiStyle& style = ImGui::GetStyle();
    std::string sdkPath = backend_->runtimeDir();
    UI::transparentTextWrapped( "%s", fmt::format( f_tr( "{} runtime not loaded:" ), backend_->name() ).c_str() );
    if ( loadCancelled_ || !error_.empty() )
        UI::transparentTextWrapped( "%s", loadCancelled_ ? _tr( "Cancelled" ) : Locale::translate( error_, Locale::genericDomain ).c_str() );
    UI::transparentTextWrapped( "%s", Locale::translate( backend_->runtimeHint(), Locale::genericDomain ).c_str() );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 1.5f * cButtonPadding * UI::scale(), cButtonPadding * UI::scale() } );
    UI::inputText( "##SdkPath", sdkPath, ImGuiInputTextFlags_ReadOnly );
    ImGui::SameLine( 0, 1.5f * style.ItemInnerSpacing.x );
    std::error_code ec;
    if ( ImGui::Link( _tr( "SDK Path" ) ) && std::filesystem::is_directory( asU8String( sdkPath ), ec ) )
        OpenDocument( asU8String( sdkPath ) );
    ImGui::PopStyleVar();
    ImGui::SameLine( 0.0f, 0.0f );
    ImGui::SameLine( 0.0f, ImGui::GetContentRegionAvail().x - 30.0f * UI::scale() );
    if ( UI::button( "...", ImVec2( 24.0f * UI::scale(), 24.0f * UI::scale() ) ) )
    {
        std::filesystem::path newPath = openFolderDialog( asU8String( sdkPath ) );
        if ( !newPath.empty() )
        {
            loadCancelled_ = false;
            loadRuntime_( utf8string( newPath ) );
        }
    }
}

void ScannerPlugin::drawClipPlane_()
{
    const ImGuiStyle& style = ImGui::GetStyle();
    const float w3 = ( ImGui::GetContentRegionAvail().x - 2 * style.ItemSpacing.x ) / 3;
    const bool hasPlane = bool( planeWidget_.getPlaneObject() );
    if ( !planeWidgetActive_ )
    {
        planeWidgetActive_ = true;
        planeWidget_.connect( &getViewerInstance(), 10, boost::signals2::at_front );
        planeWidget_.setLocalMode( true );
        if ( hasPlane )
            planeWidget_.getPlaneObject()->setVisible( true );
        updateScansBox_();
    }
    UI::transparentText( "%s", _tr( "Click and drag to draw the clip plane" ) );
    ImGui::Plane( planeWidget_, ImGui::PlaneWidgetFlags::DisableVisibility );
    const ImVec2 pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos( { pos.x, pos.y + ( cGradientButtonFramePadding - cCheckboxPadding ) * UI::scale() } );
    bool showPlane = hasPlane && planeWidget_.getPlaneObject()->isVisible();
    ImGui::BeginDisabled( !hasPlane );
    if ( UI::checkbox( _tr( "Show Plane" ), &showPlane ) )
        planeWidget_.getPlaneObject()->setVisible( showPlane );
    ImGui::EndDisabled();
    ImGui::SetCursorPos( { pos.x + w3 + style.ItemSpacing.x, pos.y } );
    if ( UI::button( ( s_tr( "Clear" ) + "##Plane" ).c_str(), hasPlane, Vector2f( w3, 0 ) ) )
        planeWidget_.undefinePlane();
    ImGui::SameLine();
    if ( UI::button( ( s_tr( "Apply" ) + "##Plane" ).c_str(), hasPlane, Vector2f( w3, 0 ) ) )
    {
        SCOPED_HISTORY( _t( "Clip scans by plane" ) );
        for ( std::shared_ptr<VisualObject> scan : validScans_() )
            clipScan( scan, planeWidget_.getPlane(), true );
        if ( streamObject_ )
            clipScan( streamObject_, planeWidget_.getPlane(), true );
    }
}

void ScannerPlugin::drawRotReg_()
{
    drawScannerSeparator( _tr( "Rotation" ) );
    UI::checkbox( _tr( "Auto rotate" ), &settings_.rotate );
    axisWidget_.setVisible( settings_.rotate );

    ImGui::BeginDisabled( !settings_.rotate );
    UI::drag<AngleUnit>( "##Angle", settings_.rotateDegrees, 1, -90, 90, { .sourceUnit = AngleUnit::degrees }, {}, 0, 0 );
    Vector2f buttonSize( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );
    ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
    if ( UI::buttonEx( "\xe2\x88\x92", buttonSize ) ) // U+2212 MINUS SIGN
        settings_.prevRotateDegrees();
    ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
    if ( UI::buttonEx( "+", buttonSize ) )
        settings_.nextRotateDegrees();
    ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
    ImGui::Text( "%s", _tr( "Angle" ) );

    buttonSize.x = -1;
    if ( UI::drag<LengthUnit>( _tr( "Pivot" ), settings_.rotateOrg ) )
    {
        axisWidget_.updateOrigin( settings_.rotateOrg );
        settings_.rotateOrgSet = true;
    }
    ImGui::SameLine( ImGui::GetContentRegionAvail().x - 64.0f * UI::scale() );
    if ( UI::button( _tr( "To center" ), buttonSize ) )
        if ( getScansCenterPoint_( settings_.rotateOrg ) )
        {
            axisWidget_.updateOrigin( settings_.rotateOrg );
            settings_.rotateOrgSet = true;
        }
    UI::setTooltipIfHovered( _tr( "Align axis in the scans center" ) );
    if ( UI::drag<LengthUnit>( _tr( "Axis" ), settings_.rotateAxis ) )
        axisWidget_.updateDirection( settings_.rotateAxis );
    ImGui::SameLine( ImGui::GetContentRegionAvail().x - 64.0f * UI::scale() );
    if ( UI::button( _tr( "To camera" ), buttonSize ) )
    {
        const Viewport& viewport = getViewerInstance().viewport();
        if ( viewport.getParameters().orthographic )
            settings_.rotateAxis = viewport.getBackwardDirection();
        else
            settings_.rotateAxis = ( viewport.getCameraPoint() - axisWidget_.getOrg() ).normalized();
        axisWidget_.updateDirection( settings_.rotateAxis );
    }
    UI::setTooltipIfHovered( _tr( "Orient axis to current camera direction" ) );

    buttonSize.x = ( ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x ) / 2;
    if ( UI::button( !axisWidget_.isPickMode() ? _tr( "Pick" ) : _tr( "Picking..." ), buttonSize ) )
        axisWidget_.setPickMode( !axisWidget_.isPickMode() );
    UI::setTooltipIfHovered( _tr( "Pick a point from scene (hold Shift to set direction)" ) );
    ImGui::SameLine();
    if ( UI::button( _tr( "Save" ), buttonSize ) )
        settings_.save( configKey_(), true );
    UI::setTooltipIfHovered( _tr( "Save rotation parameters" ) );
    ImGui::EndDisabled();

    drawScannerSeparator( _tr( "Registration" ) );
    UI::checkbox( _tr( "Auto register" ), &settings_.registration );
    UI::setTooltipIfHovered( _tr( "Automatically align a new scan to previous scans" ) );
    UI::checkboxValid( _tr( "Try RANSAC initial position" ), &icp_.ransac, settings_.registration && icp_.rigidMode() );
    UI::setTooltipIfHovered( _tr( "Also align a new scan from any position by matching shape descriptors (RANSAC), then ICP, "
        "and keep that pose if it has at least a quarter more samples within Sampling Spacing of the other surface than ICP alone. "
        "Needs normals and the Rigid Body or Rigid + Scale Transformation" ) );
    if ( RibbonButtonDrawer::CustomCollapsingHeader( _tr( "ICP Parameters" ) ) )
    {
        ICPDialog::drawPairsFormingSettings( icp_.params, icp_.samplingVoxelSize, icp_.objectSize );
        ICPDialog::drawStopConditions( icp_.params );
        if ( icp_.info )
            ICPDialog::drawInformation( icp_.info->numSamples, icp_.info->numPairs, icp_.info->meanSqDistToPoint, icp_.info->meanSqDistToPlane );
        if ( !icp_.result.empty() )
        {
            ImGui::Separator();
            ICPDialog::drawResult( icp_.result );
        }
    }
}

void ScannerPlugin::drawPreview_()
{
    if ( imagesDirty_ )
        refreshPreview_();
    if ( !hasImage_() )
        return;
    drawScannerSeparator( _tr( "Preview" ) );
    if ( isWindowOpen_ )
    {
        const float w3 = ( ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x ) / 3;
        UI::alignTextToButton();
        ImGui::TextUnformatted( _tr( "Preview window popped out" ) );
        ImGui::SameLine();
        if ( UI::button( _tr( "Close" ), Vector2f( w3, 0 ) ) )
            isWindowOpen_ = false;
        return;
    }
    ImVec2 size = imageSize_( ImGui::GetContentRegionAvail().x );
    if ( size.y > size.x * 1.1f )
        size.y /= 2;
    float y = ImGui::GetCursorPosY();
    drawImages_( size );
    if ( ImGui::IsItemHovered() )
    {
        ImVec2 savePos = ImGui::GetCursorPos();
        ImGui::SetCursorPos( { ImGui::GetContentRegionAvail().x - 64.0f * UI::scale(), y } );
        if ( UI::button( _tr( "Pop Out" ), Vector2f( -1, 0 ) ) || ImGui::IsItemClicked() )
            isWindowOpen_ = true;
        ImGui::SetCursorPos( savePos );
    }
}

void ScannerPlugin::drawPreviewWindow_()
{
    if ( !isWindowOpen_ )
        return;
    if ( imagesDirty_ )
        refreshPreview_();
    if ( !hasImage_() ) // e.g. never for a backend without 2D images
        return;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 frameSize{ ( float )getViewerInstance().framebufferSize.x, ( float )getViewerInstance().framebufferSize.y };
    if ( windowSize_.x == 0 )
        windowSize_.x = frameSize.x / 3;
    ImVec2 size = imageSize_( windowSize_.x - 2 * style.WindowPadding.x );
    if ( const float maxHeight = 1024.0f * UI::scale(); size.y > maxHeight )
        size = { floor( size.x * maxHeight / size.y ), maxHeight };
    windowSize_.y = size.y + 2 * ( style.WindowPadding.y + 2 * style.ItemSpacing.y );
    ImGui::SetNextWindowSize( windowSize_, ImGuiCond_Always );
    const float minWidth = 360.0f * UI::scale();
    ImGui::SetNextWindowSizeConstraints( { minWidth, minWidth / 2 }, frameSize );
    ImGui::SetNextWindowPos( { frameSize.x / 3, frameSize.y / 5 }, ImGuiCond_Once );
    if ( ImGui::Begin( ( name() + " Preview" ).c_str(), &isWindowOpen_, 0 ) )
    {
        windowSize_.x = ImGui::GetWindowSize().x;
        drawImages_( size );
    }
    ImGui::End();
}

void ScannerPlugin::drawDialog( ImGuiContext* )
{
    const float menuWidth = 372.0f * UI::scale();
    if ( ImGuiBeginWindow_( { .width = menuWidth } ) )
    {
        drawBackendCombo_(); // also above the not-loaded page, to leave a backend without its SDK
        if ( !backend_->runtimeLoaded() )
        {
            axisWidget_.setVisible( false );
            isWindowOpen_ = false;
            drawNotLoaded_();
        }
        else
        {
            const ScannerOnScan onScan = [this]( std::shared_ptr<VisualObject> obj ) { onScanProduced_( obj ); };
            scanTabActive_ = false;
            if ( UI::beginTabBar( "##MainTabs" ) )
            {
                if ( UI::beginTabItem( _tr( "Parameters" ), nullptr, selectFirstTab_ ? ImGuiTabItemFlags_SetSelected : 0 ) )
                {
                    drawScannerParams( *backend_ );
                    const Vector2f buttonSize3( ( ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x ) / 3, 0 );
                    ImGui::Dummy( buttonSize3 );
                    ImGui::SameLine();
                    if ( UI::button( _tr( "Reset" ), buttonSize3 ) )
                    {
                        backend_->resetSettings();
                        settings_ = {};
                        axisWidget_.updateRay( { settings_.rotateOrg, settings_.rotateAxis } );
                    }
                    ImGui::SameLine();
                    if ( UI::button( _tr( "Save defaults" ), buttonSize3 ) )
                    {
                        backend_->saveSettings();
                        settings_.save( configKey_() );
                    }
                    UI::setTooltipIfHovered( _tr( "Save scanner and processing parameters as default" ) );
                    paramsEndY_ = ImGui::GetCursorPosY();
                    UI::endTabItem();
                }
                selectFirstTab_ = false;
                if ( UI::beginTabItem( _tr( "Clip Plane" ) ) )
                {
                    drawClipPlane_();
                    UI::endTabItem();
                }
                else if ( planeWidgetActive_ )
                {
                    // the plane widget takes mouse drags in the viewport only while its tab is open
                    planeWidgetActive_ = false;
                    planeWidget_.disconnect();
                    planeWidget_.setOnPlaneUpdateCallback( nullptr );
                    if ( planeWidget_.getPlaneObject() )
                        planeWidget_.getPlaneObject()->setVisible( false );
                }
                if ( UI::beginTabItem( _tr( "Rot/Reg" ) ) )
                {
                    drawRotReg_();
                    UI::endTabItem();
                }
                else
                    axisWidget_.setVisible( false );
                // the backend's actions and the preview of their results
                if ( backend_->capabilities().actions && UI::beginTabItem( _tr( "Current Scan" ) ) )
                {
                    scanTabActive_ = true;
                    if ( drawScannerActions( *backend_, onScan, error_ ) )
                        imagesDirty_ = true; // the held raw scan changed
                    drawPreview_();
                    UI::endTabItem();
                }
                UI::endTabBar();
            }
            ImGui::SetCursorPosY( std::max( paramsEndY_, ImGui::GetCursorPosY() ) );
            UI::highlightWindowArea(); // shade the capture block apart from the tabs
            if ( drawScannerCapture( *backend_, onScan, error_ ) )
                imagesDirty_ = true; // even a failed capture may have dropped the held raw scan

            const bool live = backend_->livePreviewActive();
            if ( live && !wasLive_ && !scanTabActive_ )
                isWindowOpen_ = true;
            if ( !live && wasLive_ )
                clearStream_();
            wasLive_ = live;
        }
        ImGui::EndCustomStatePlugin();
    }
    drawPreviewWindow_();
}

} // namespace

std::shared_ptr<RibbonMenuItem> makeScannerPlugin( std::vector<std::unique_ptr<IScannerBackend>> backends )
{
    return backends.empty() ? nullptr : std::make_shared<ScannerPlugin>( std::move( backends ) );
}

} // namespace MR
