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

    /// include one more face in the final forest, returning the newly reached face;
    /// returns invalid FaceId if no more faces are reachable
    FaceId reachNext();

    /// adds steps for all faces around reached face (f);
    /// returns true if at least one step was added
    bool addRingSteps( FaceId f );

    /// the same as reachNext() + addRingSteps()
    FaceId growOneEdge();

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
    nextSteps_ = { std::less<CandidateFace>(), std::move( queueData ) };
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

bool DualEdgePathsBuider::addRingSteps( FaceId f )
{
    bool aNextStepAdded = false;
    if ( !f )
        return aNextStepAdded;
    const float fDist = distances_[f];
    for ( EdgeId e : leftRing( topology_, f ) )
    {
        CandidateFace c;
        c.f = topology_.right( e );
        if ( !c.f )
            continue;
        c.penalty = fDist + metric_( e );
        aNextStepAdded = addNextStep_( c ) || aNextStepAdded;
    }
    return aNextStepAdded;
}

FaceId DualEdgePathsBuider::reachNext()
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
        return c.f;
    }
    return {};
}


FaceId DualEdgePathsBuider::growOneEdge()
{
    auto res = reachNext();
    addRingSteps( res );
    return res;
}

} // anonymous namespace

FaceScalars calcFaceDistances( const MeshTopology & topology, const EdgeMetric & metric, const FaceBitSet & starts )
{
    MR_TIMER
    DualEdgePathsBuider builder( topology, metric, starts );
    while ( builder.growOneEdge() )
        {}
    return builder.takeDistances();
}

} // namespace MR
