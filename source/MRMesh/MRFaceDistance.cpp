#include "MRFaceDistance.h"
#include "MRMeshTopology.h"
#include "MRExpandShrink.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include <cfloat>
#include <queue>

namespace MR
{

namespace
{

class DualEdgePathsBuider
{
public:
    /// see calcFaceDistances(...) for the description of the parameters
    DualEdgePathsBuider( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts );

    struct CandidateFace
    {
        FaceId f;
        // best penalty to reach this face
        float penalty = FLT_MAX;

        // smaller penalty to be the first
        friend bool operator <( const CandidateFace & a, const CandidateFace & b )
        {
            return a.penalty > b.penalty;
        }
    };

    /// include one more face in the final forest, returning the newly reached face;
    /// returns invalid FaceId if no more faces are reachable
    CandidateFace reachNext();

    /// adds steps for all faces around reached face (f);
    /// returns true if at least one step was added
    bool addRingSteps( const CandidateFace & c );

    /// the same as reachNext() + addRingSteps()
    CandidateFace growOneEdge();

    /// returns true if no more reachable faces are available
    bool done() const { return nextSteps_.empty(); }

    /// returns path length till the next candidate face or maximum float value if all faces have been reached
    float doneDistance() const { return nextSteps_.empty() ? FLT_MAX : nextSteps_.top().penalty; }

    /// gives read access to the distances found so far
    const FaceScalars & distances() const { return distances_; }

    /// extracts computed distances when the building is finished
    FaceScalars takeDistances() { return std::move( distances_ ); }

private:
    const MeshTopology & topology_;
    EdgeMetric metric_;
    FaceScalars distances_;

    std::priority_queue<CandidateFace> nextSteps_;

    // compares proposed step with the value known for this face;
    // if proposed step is smaller then adds it in the queue and returns true;
    // otherwise if the known metric to org( c.back ) is already not greater than returns false
    bool addNextStep_( const CandidateFace & c );
};

DualEdgePathsBuider::DualEdgePathsBuider( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts )
    : topology_( topology )
    , metric_( metric )
{
    distances_.resize( topology.faceSize(), FLT_MAX );
    for ( FaceId s : starts )
        distances_[s] = 0;

    auto bdFaces = getBoundaryFaces( topology, starts );
    std::vector<CandidateFace> queueData;
    queueData.reserve( bdFaces.count() );
    for ( FaceId b : bdFaces )
        queueData.push_back( { b, 0 } );
    nextSteps_ = std::priority_queue<CandidateFace>{ std::less<CandidateFace>(), std::move( queueData ) };
}

bool DualEdgePathsBuider::addNextStep_( const CandidateFace & c )
{
    if ( !( c.penalty < FLT_MAX ) )
        return false; // maximal or infinity metric means that this path shall be skipped
    if ( distances_[c.f] > c.penalty )
    {
        distances_[c.f] = c.penalty;
        nextSteps_.push( c );
        return true;
    }
    return false;
}

bool DualEdgePathsBuider::addRingSteps( const CandidateFace & c )
{
    bool aNextStepAdded = false;
    if ( !c.f )
        return aNextStepAdded;
    assert( c.penalty == distances_[c.f] );
    for ( EdgeId e : leftRing( topology_, c.f ) )
    {
        CandidateFace n;
        n.f = topology_.right( e );
        if ( !n.f )
            continue;
        n.penalty = c.penalty + metric_( e );
        aNextStepAdded = addNextStep_( n ) || aNextStepAdded;
    }
    return aNextStepAdded;
}

auto DualEdgePathsBuider::reachNext() -> CandidateFace
{
    while ( !nextSteps_.empty() )
    {
        const auto c = nextSteps_.top();
        nextSteps_.pop();
        if ( distances_[c.f] < c.penalty )
        {
            // shorter path to the face was found
            continue;
        }
        assert( distances_[c.f] == c.penalty );
        return c;
    }
    return {};
}


auto DualEdgePathsBuider::growOneEdge() -> CandidateFace
{
    auto res = reachNext();
    addRingSteps( res );
    return res;
}

} // anonymous namespace

std::optional<FaceScalars> calcFaceDistances( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts,
    const FaceDistancesSettings & settings )
{
    MR_TIMER
    DualEdgePathsBuider builder( topology, metric, starts );
    float localMaxDist = 0;
    size_t numDone = settings.progress ? starts.count() : 0;
    const float rTotal = 1.0f / topology.numValidFaces();
    if ( !reportProgress( settings.progress, numDone * rTotal ) )
        return {};

    FaceScalars order;
    if ( settings.out == FaceDistancesSettings::OutputFaceValues::SeqOrder )
        order = builder.distances();
    int lastSeq = 0;
    for (;;)
    {
        const auto c = builder.growOneEdge();
        if ( !c.f )
            break;
        localMaxDist = c.penalty;
        if ( settings.out == FaceDistancesSettings::OutputFaceValues::SeqOrder && !starts.test( c.f ) )
            order[c.f] = float( ++lastSeq );
        if ( !reportProgress( settings.progress, [&]() { return numDone * rTotal; }, ++numDone, 16384 ) )
            return {};
    }
    if ( settings.out == FaceDistancesSettings::OutputFaceValues::SeqOrder )
    {
        if ( settings.maxDist )
            *settings.maxDist = float( lastSeq );
        return order;
    }
    if ( settings.maxDist )
        *settings.maxDist = localMaxDist;
    return builder.takeDistances();
}

} // namespace MR
