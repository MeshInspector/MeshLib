#include "MRMeshDecimate.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRCylinder.h"
#include "MRGTest.h"
#include "MRMeshDelone.h"
#include "MRMeshSubdivide.h"
#include "MRPch/MRTBB.h"
#include <queue>

namespace MR
{

// collapses given edge and deletes
// 1) faces: left( e ) and right( e );
// 2) vertex org( e )/dest( e ) if given edge was their only edge, otherwise only dest( e );
// 3) edges: e, next( e.sym() ), prev( e.sym() );
// returns prev( e ) if it is valid
EdgeId collapseEdge( MeshTopology & topology, const EdgeId e )
{
    topology.setLeft( e, FaceId() );
    topology.setLeft( e.sym(), FaceId() );

    if ( topology.next( e ) == e )
    {
        topology.setOrg( e, VertId() );
        const EdgeId b = topology.prev( e.sym() );
        if ( b == e.sym() )
            topology.setOrg( e.sym(), VertId() );
        else
            topology.splice( b, e.sym() );

        assert( topology.isLoneEdge( e ) );
        return EdgeId();
    }

    topology.setOrg( e.sym(), VertId() );

    const EdgeId ePrev = topology.prev( e );
    const EdgeId eNext = topology.next( e );
    if ( ePrev != e )
        topology.splice( ePrev, e );

    const EdgeId a = topology.next( e.sym() );
    if ( a == e.sym() )
    {
        assert( topology.isLoneEdge( e ) );
        return ePrev != e ? ePrev : EdgeId();
    }
    const EdgeId b = topology.prev( e.sym() );

    topology.splice( b, e.sym() );
    assert( topology.isLoneEdge( e ) );

    assert( topology.next( b ) == a );
    assert( topology.next( ePrev ) == eNext );
    topology.splice( b, ePrev );
    assert( topology.next( b ) == eNext );
    assert( topology.next( ePrev ) == a );

    if ( topology.next( a.sym() ) == ePrev.sym() )
    {
        topology.splice( ePrev, a );
        topology.splice( topology.prev( a.sym() ), a.sym() );
        assert( topology.isLoneEdge( a ) );
        if ( !topology.left( ePrev ) && !topology.right( ePrev ) )
        {
            topology.splice( topology.prev( ePrev ), ePrev );
            topology.splice( topology.prev( ePrev.sym() ), ePrev.sym() );
            topology.setOrg( ePrev, {} );
            topology.setOrg( ePrev.sym(), {} );
        }
    }

    if ( topology.next( eNext.sym() ) == b.sym() )
    {
        topology.splice( eNext.sym(), b.sym() );
        topology.splice( topology.prev( b ), b );
        assert( topology.isLoneEdge( b ) );
        if ( !topology.left( eNext ) && !topology.right( eNext ) )
        {
            topology.splice( topology.prev( eNext ), eNext );
            topology.splice( topology.prev( eNext.sym() ), eNext.sym() );
            topology.setOrg( eNext, {} );
            topology.setOrg( eNext.sym(), {} );
        }
    }

    return ePrev != e ? ePrev : EdgeId();
}

class MeshDecimator
{
public:
    MeshDecimator( Mesh & mesh, const DecimateSettings & settings );
    DecimateResult run();

    // returns true if the collapse of given edge is permitted by the region and settings
    bool isInRegion( EdgeId e ) const;

private: 
    Mesh & mesh_;
    const DecimateSettings & settings_;
    const float maxErrorSq_;
    Vector<QuadraticForm3f, VertId> vertForms_;
    struct QueueElement
    {
        float c = 0;
        UndirectedEdgeId uedgeId;
        std::pair<float, UndirectedEdgeId> asPair() const { return { -c, uedgeId }; }
        bool operator < ( const QueueElement & r ) const { return asPair() < r.asPair(); }
    };
    std::priority_queue<QueueElement> queue_;
    UndirectedEdgeBitSet presentInQueue_;
    DecimateResult res_;
    std::vector<VertId> originNeis_;
    std::vector<Vector3f> triDblAreas_; // directed double areas of newly formed triangles to check that they are consistently oriented
    class EdgeMetricCalc;

    bool initializeQueue_();
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm = nullptr, Vector3f * outCollapsePos = nullptr ) const;
    void addInQueueIfMissing_( UndirectedEdgeId ue );
    VertId collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos );
};

MeshDecimator::MeshDecimator( Mesh & mesh, const DecimateSettings & settings )
    : mesh_( mesh )
    , settings_( settings )
    , maxErrorSq_( sqr( settings.maxError ) )
{
}

bool MeshDecimator::isInRegion( EdgeId e ) const
{
    if ( !mesh_.topology.isInnerOrBdEdge( e, settings_.region ) )
        return false;
    if ( !settings_.touchBdVertices )
    {
        if ( mesh_.topology.isBdVertexInOrg( e, settings_.region ) )
            return false;
        if ( mesh_.topology.isBdVertexInOrg( e.sym(), settings_.region ) )
            return false;
    }
    return true;
}

class MeshDecimator::EdgeMetricCalc 
{
public:
    EdgeMetricCalc( const MeshDecimator & decimator ) : decimator_( decimator ) { }
    EdgeMetricCalc( EdgeMetricCalc & x, tbb::split ) : decimator_( x.decimator_ ) { }
    void join( EdgeMetricCalc & y ) { auto yes = y.takeElements(); elems_.insert( elems_.end(), yes.begin(), yes.end() ); }

    const std::vector<QueueElement> & elements() const { return elems_; }
    std::vector<QueueElement> takeElements() { return std::move( elems_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r ) 
    {
        const auto & mesh = decimator_.mesh_;
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            EdgeId e{ ue };
            if ( mesh.topology.isLoneEdge( e ) )
                continue;
            if ( !decimator_.isInRegion( e ) )
                continue;
            if ( auto qe = decimator_.computeQueueElement_( ue ) )
                elems_.push_back( *qe );
        }
    }
            
public:
    const MeshDecimator & decimator_;
    std::vector<QueueElement> elems_;
};

MR::QuadraticForm3f computeFormAtVertex( const MR::MeshPart & mp, MR::VertId v, float stabilizer )
{
    QuadraticForm3f qf;
    qf.addDistToOrigin( stabilizer );
    for ( EdgeId e : orgRing( mp.mesh.topology, v ) )
    {
        if ( mp.mesh.topology.isBdEdge( e, mp.region ) )
            qf.addDistToLine( mp.mesh.edgeVector( e ).normalized() );
        if ( mp.mesh.topology.isLeftInRegion( e, mp.region ) )
            qf.addDistToPlane( mp.mesh.leftNormal( e ) );
    }
    return qf;
}

bool resolveMeshDegenerations( MR::Mesh& mesh, int maxIters, float maxDeviation, float maxAngleChange, float criticalAspectRatio )
{
    MR_TIMER;
    bool meshChanged = false;
    for( int i = 0; i < maxIters; ++i )
    {
        DeloneSettings delone
        {
            .maxDeviationAfterFlip = maxDeviation,
            .maxAngleChange = maxAngleChange,
            .criticalTriAspectRatio = criticalAspectRatio
        };
        bool changedThisIter = makeDeloneEdgeFlips( mesh, delone, 5 ) > 0;

        DecimateSettings settings;
        settings.maxError = maxDeviation;
        changedThisIter = decimateMesh( mesh, settings ).vertsDeleted > 0 || changedThisIter;
        meshChanged = meshChanged || changedThisIter;
        if ( !changedThisIter )
            break;
    }
    return meshChanged;
}

bool MeshDecimator::initializeQueue_()
{
    MR_TIMER;

    VertBitSet store;
    const VertBitSet & regionVertices = getIncidentVerts( mesh_.topology, settings_.region, store );

    if ( settings_.vertForms && !settings_.vertForms->empty() )
    {
        assert( settings_.vertForms->size() > mesh_.topology.lastValidVert() );
        vertForms_ = std::move( *settings_.vertForms );
    }
    else
    {
        vertForms_.resize( mesh_.topology.lastValidVert() + 1 );
        BitSetParallelFor( regionVertices, [&]( VertId v )
        {
            vertForms_[v] = computeFormAtVertex( MR::MeshPart{ mesh_, settings_.region }, v, settings_.stabilizer );
        } );
    }

    if ( settings_.progressCallback && !settings_.progressCallback( 0.1f ) )
        return false;

    EdgeMetricCalc calc( *this );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( UndirectedEdgeId{0}, UndirectedEdgeId{mesh_.topology.undirectedEdgeSize()} ), calc );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.2f ) )
        return false;

    presentInQueue_.resize( mesh_.topology.undirectedEdgeSize() );
    for ( const auto & qe : calc.elements() )
        presentInQueue_.set( qe.uedgeId );
    queue_ = std::priority_queue<QueueElement>{ std::less<QueueElement>(), calc.takeElements() };

    if ( settings_.progressCallback && !settings_.progressCallback( 0.25f ) )
        return false;
    return true;
}

auto MeshDecimator::computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm, Vector3f * outCollapsePos ) const -> std::optional<QueueElement>
{
    EdgeId e{ ue };
    const auto o = mesh_.topology.org( e );
    const auto d = mesh_.topology.org( e.sym() );
    const auto po = mesh_.points[o];
    const auto pd = mesh_.points[d];
    const auto vo = vertForms_[o];
    const auto vd = vertForms_[d];
    if ( ( po - pd ).lengthSq() > sqr( settings_.maxEdgeLen ) )
        return {};

    QueueElement res;
    res.uedgeId = ue;

    if ( settings_.strategy == DecimateStrategy::ShortestEdgeFirst )
    {
        res.c = mesh_.edgeLengthSq( e );
        if ( !settings_.adjustCollapse && res.c > maxErrorSq_ )
            return {};
    }

    QuadraticForm3f qf;
    Vector3f pos;
    std::tie( qf, pos ) = sum( vo, po, vd, pd, !settings_.optimizeVertexPos );

    if ( settings_.strategy == DecimateStrategy::MinimizeError )
    {
        if ( !settings_.adjustCollapse && qf.c > maxErrorSq_ )
            return {};
        res.c = qf.c;
    }

    if ( settings_.adjustCollapse )
    {
        const auto pos0 = pos;
        settings_.adjustCollapse( ue, res.c, pos );
        if ( res.c > maxErrorSq_ )
            return {};
        if ( outCollapseForm && pos != pos0 )
            qf.c = vo.eval( po - pos ) + vd.eval( pd - pos );
    }

    if ( outCollapseForm )
        *outCollapseForm = qf;
    if ( outCollapsePos )
        *outCollapsePos = pos;

    return res;
}

void MeshDecimator::addInQueueIfMissing_( UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    if ( !isInRegion( e ) )
        return;
    if ( presentInQueue_.test_set( ue ) )
        return;
    if ( auto qe = computeQueueElement_( ue ) )
        queue_.push( *qe );
}

VertId MeshDecimator::collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos )
{
    auto & topology = mesh_.topology;
    // cannot collapse edge if its left and right faces share another edge
    if ( auto pe = topology.prev( edgeToCollapse ); pe != edgeToCollapse && pe == topology.next( edgeToCollapse ) )
        return {};
    if ( auto pe = topology.prev( edgeToCollapse.sym() ); pe != edgeToCollapse.sym() && pe == topology.next( edgeToCollapse.sym() ) )
        return {};

    auto vo = topology.org( edgeToCollapse );
    auto vd = topology.dest( edgeToCollapse );
    auto po = mesh_.points[vo];
    auto pd = mesh_.points[vd];
    if ( !settings_.optimizeVertexPos && collapsePos == pd )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        std::swap( vo, vd );
        std::swap( po, pd );
    }
    auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

    auto dirDblArea = [&]( VertId nei1, VertId nei2 )
    {
        const auto pos1 = mesh_.points[nei1];
        const auto pos2 = mesh_.points[nei2];
        return cross( pos1 - collapsePos, pos2 - collapsePos );
    };
    auto triangleAspectRatio = [&]( const Vector3f & v0, VertId nei1, VertId nei2 )
    {
        return MR::triangleAspectRatio( v0, mesh_.points[nei1], mesh_.points[nei2] );
    };

    float maxOldAspectRatio = settings_.maxTriangleAspectRatio;
    float maxNewAspectRatio = 0;

    originNeis_.clear();
    triDblAreas_.clear();
    Vector3d sumDblArea_;
    for ( EdgeId e : orgRing0( topology, edgeToCollapse ) )
    {
        auto eDest = topology.dest( e );
        auto eDest2 = topology.dest( topology.next( e ) );
        if ( eDest == vd )
            return {}; // multiple edge found
        if ( eDest != vl && eDest != vr )
            originNeis_.push_back( eDest );
        if ( eDest != vr && topology.left( e ) )
        {
            auto da = dirDblArea( eDest, eDest2 );
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            maxNewAspectRatio = std::max( maxNewAspectRatio, triangleAspectRatio( collapsePos, eDest, eDest2 ) );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( po, eDest, eDest2 ) );
    }
    std::sort( originNeis_.begin(), originNeis_.end() );

    for ( EdgeId e : orgRing0( topology, edgeToCollapse.sym() ) )
    {
        auto eDest = topology.dest( e );
        auto eDest2 = topology.dest( topology.next( e ) );
        assert ( eDest != vo );
        if ( std::binary_search( originNeis_.begin(), originNeis_.end(), eDest ) )
            return {}; // to prevent appearance of multiple edges
        if ( eDest != vl && topology.left( e ) )
        {
            auto da = dirDblArea( eDest, eDest2 );
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            maxNewAspectRatio = std::max( maxNewAspectRatio, triangleAspectRatio( collapsePos, eDest, eDest2 ) );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( pd, eDest, eDest2 ) );
    }

    if ( maxNewAspectRatio > maxOldAspectRatio )
        return {}; // new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings

    // checks that all new normals are consistent (do not check for degenerate edges)
    if ( ( po != pd ) || ( po != collapsePos ) )
    {
        auto n = Vector3f{ sumDblArea_.normalized() };
        for ( const auto da : triDblAreas_ )
            if ( dot( da, n ) < 0 )
                return {};
    }

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return {}; // user prohibits the collapse

    ++res_.vertsDeleted;
    if ( vl )
        ++res_.facesDeleted;
    if ( vr )
        ++res_.facesDeleted;

    mesh_.points[vo] = collapsePos;
    if ( settings_.region )
    {
        if ( auto l = topology.left( edgeToCollapse ) )
            settings_.region->reset( l );
        if ( auto r = topology.left( edgeToCollapse.sym() ) )
            settings_.region->reset( r );
    }
    collapseEdge( topology, edgeToCollapse );
    return topology.hasVert( vo ) ? vo : VertId{};
}

DecimateResult MeshDecimator::run()
{
    MR_TIMER;

    if ( !initializeQueue_() )
        return res_;

    res_.errorIntroduced = settings_.maxError;
    int lastProgressFacesDeleted = 0;
    const int maxFacesDeleted = std::min(
        settings_.region ? (int)settings_.region->count() : mesh_.topology.numValidFaces(), settings_.maxDeletedFaces );
    while ( !queue_.empty() )
    {
        auto topQE = queue_.top();
        assert( presentInQueue_.test( topQE.uedgeId ) );
        queue_.pop();
        if ( res_.facesDeleted >= settings_.maxDeletedFaces || res_.vertsDeleted >= settings_.maxDeletedVertices )
        {
            res_.errorIntroduced = std::sqrt( topQE.c );
            break;
        }

        if ( settings_.progressCallback && res_.facesDeleted >= 1000 + lastProgressFacesDeleted ) 
        {
            if ( !settings_.progressCallback( 0.25f + 0.75f * res_.facesDeleted / maxFacesDeleted ) )
                return res_;
            lastProgressFacesDeleted = res_.facesDeleted;
        }

        if ( mesh_.topology.isLoneEdge( topQE.uedgeId ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( topQE.uedgeId );
            continue;
        }

        QuadraticForm3f collapseForm;
        Vector3f collapsePos;
        auto qe = computeQueueElement_( topQE.uedgeId, &collapseForm, &collapsePos );
        if ( !qe )
        {
            presentInQueue_.reset( topQE.uedgeId );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            continue;
        }

        presentInQueue_.reset( topQE.uedgeId );
        VertId collapseVert = collapse_( topQE.uedgeId, collapsePos );
        if ( !collapseVert )
            continue;

        vertForms_[collapseVert] = collapseForm;

        for ( EdgeId e : orgRing( mesh_.topology, collapseVert ) )
        {
            addInQueueIfMissing_( e.undirected() );
            if ( mesh_.topology.left( e ) )
                addInQueueIfMissing_( mesh_.topology.prev( e.sym() ).undirected() );
        }
    }

    if ( settings_.packMesh )
    {
        FaceMap fmap;
        VertMap vmap;
        mesh_.pack( 
            settings_.region ? &fmap : nullptr,
            settings_.vertForms ? &vmap : nullptr );

        if ( settings_.region )
            *settings_.region = settings_.region->getMapping( fmap, mesh_.topology.faceSize() );

        if ( settings_.vertForms )
        {
            for ( VertId oldV{ 0 }; oldV < vmap.size(); ++oldV )
                if ( auto newV = vmap[oldV] )
                    if ( newV < oldV )
                        vertForms_[newV] = vertForms_[oldV];
            vertForms_.resize( mesh_.topology.vertSize() );
        }
    }

    if ( settings_.vertForms )
        *settings_.vertForms = std::move( vertForms_ );
    res_.cancelled = false;
    return res_;
}

DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER;
    MR_WRITER( mesh );
    MeshDecimator md( mesh, settings );
    return md.run();
}

bool remesh( MR::Mesh& mesh, const RemeshSettings & settings )
{
    MR_TIMER;
    MR_WRITER( mesh );

    if ( settings.progressCallback && !settings.progressCallback( 0.0f ) )
        return false;

    SubdivideSettings subs;
    subs.maxEdgeLen = 2 * settings.targetEdgeLen;
    subs.maxEdgeSplits = 10'000'000;
    subs.maxAngleChangeAfterFlip = settings.maxAngleChangeAfterFlip;
    subs.useCurvature = settings.useCurvature;
    subs.region = settings.region;
    subs.notFlippable = settings.notFlippable;
    subs.onEdgeSplit = settings.onEdgeSplit;
    if ( settings.progressCallback )
        subs.progressCallback = [settings] ( float arg ) { return settings.progressCallback( arg * 0.5f ); };
    subdivideMesh( mesh, subs );

    if ( settings.progressCallback && !settings.progressCallback( 0.5f ) )
        return false;

    DecimateSettings decs;
    decs.strategy = DecimateStrategy::ShortestEdgeFirst;
    decs.maxError = settings.targetEdgeLen / 2;
    decs.region = settings.region;
    decs.packMesh = settings.packMesh;
    if ( settings.progressCallback )
        decs.progressCallback = [settings] ( float arg ) { return settings.progressCallback( 0.5f + arg * 0.5f ); };
    decimateMesh( mesh, decs );

    if ( settings.progressCallback && !settings.progressCallback( 1.0f ) )
        return false;

    return true;
}

// check if Decimator updates region
TEST( MRMesh, MeshDecimate )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettings decimateSettings;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    auto decimateResults = decimateMesh(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);
}

} //namespace MR
