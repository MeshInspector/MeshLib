#include "MREMeshDecimate.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRQuadraticForm.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRTriMath.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMeshDelone.h"
#include "MRPch/MRTBB.h"
#include <queue>

namespace MRE
{

using namespace MR;

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
    }

    if ( topology.next( eNext.sym() ) == b.sym() )
    {
        topology.splice( eNext.sym(), b.sym() );
        topology.splice( topology.prev( b ), b );
        assert( topology.isLoneEdge( b ) );
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

    void initializeQueue_();
    QueueElement computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm = nullptr, Vector3f * outCollapsePos = nullptr ) const;
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
            auto qe = decimator_.computeQueueElement_( ue );
            if ( qe.c <= decimator_.maxErrorSq_ )
                elems_.push_back( qe );
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

bool resolveMeshDegenerations( MR::Mesh& mesh, int maxIters, float maxDeviation )
{
    MR_TIMER;
    bool meshChanged = false;
    for( int i = 0; i < maxIters; ++i )
    {
        bool changedThisIter = makeDeloneEdgeFlips( mesh, 5, maxDeviation ) > 0;

        DecimateSettings settings;
        settings.maxError = maxDeviation;
        changedThisIter = decimateMesh( mesh, settings ).vertsDeleted > 0 || changedThisIter;
        meshChanged = meshChanged || changedThisIter;
        if ( !changedThisIter )
            break;
    }
    return meshChanged;
}

void MeshDecimator::initializeQueue_()
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

    EdgeMetricCalc calc( *this );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( UndirectedEdgeId{0}, UndirectedEdgeId{mesh_.topology.undirectedEdgeSize()} ), calc );
    presentInQueue_.resize( mesh_.topology.undirectedEdgeSize() );
    for ( const auto & qe : calc.elements() )
        presentInQueue_.set( qe.uedgeId );
    queue_ = std::priority_queue<QueueElement>{ std::less<QueueElement>(), calc.takeElements() };
}

auto MeshDecimator::computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm, Vector3f * outCollapsePos ) const -> QueueElement
{
    QueueElement res;
    res.uedgeId = ue;
    EdgeId e{ ue };
    auto o = mesh_.topology.org( e );
    auto d = mesh_.topology.org( e.sym() );
    auto [qf, pos] = sum( vertForms_[o], mesh_.points[o], vertForms_[d], mesh_.points[d] );
    res.c = qf.c;
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
    auto qe = computeQueueElement_( ue );
    if ( qe.c <= maxErrorSq_ )
        queue_.push( qe );
}

VertId MeshDecimator::collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos )
{
    auto & topology = mesh_.topology;
    const auto vo = topology.org( edgeToCollapse );
    const auto vd = topology.dest( edgeToCollapse );
    const auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    const auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

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
    const Vector3f oldOrig = mesh_.points[vo];
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
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( oldOrig, eDest, eDest2 ) );
    }
    std::sort( originNeis_.begin(), originNeis_.end() );

    const Vector3f oldDest = mesh_.points[vd];
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
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( oldDest, eDest, eDest2 ) );
    }

    if ( maxNewAspectRatio > maxOldAspectRatio )
        return {}; // new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings

    // checks that all new normals are consistent (do not check for degenerate edges)
    if ( ( oldOrig != oldDest ) || ( oldOrig != collapsePos ) )
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

    initializeQueue_();

    res_.errorIntroduced = settings_.maxError;
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

        if ( mesh_.topology.isLoneEdge( topQE.uedgeId ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( topQE.uedgeId );
            continue;
        }

        QuadraticForm3f collapseForm;
        Vector3f collapsePos;
        auto qe = computeQueueElement_( topQE.uedgeId, &collapseForm, &collapsePos );

        if ( qe.c > topQE.c )
        {
            if ( qe.c <= maxErrorSq_ )
                queue_.push( qe );
            else
                presentInQueue_.reset( topQE.uedgeId );
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

    if ( settings_.vertForms )
        *settings_.vertForms = std::move( vertForms_ );
    return res_;
}

DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER;
    MR_MESH_WRITER( mesh );
    MeshDecimator md( mesh, settings );
    return md.run();
}


// check if Decimator updates region
TEST(MREAlgorithms, MeshDecimate)
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
