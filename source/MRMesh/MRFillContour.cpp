#include "MRFillContour.h"
#include "MRMeshTopology.h"
#include "MRBitSet.h"
#include "MRphmap.h"
#include "MRTimer.h"

namespace MR
{

class ContourLeftFiller
{
public:
    ContourLeftFiller( const MeshTopology & topology );
    void addContour( const std::vector<EdgeId> & contour );
    const FaceBitSet& fill();

private:
    const MeshTopology & topology_;
    FaceBitSet filledFaces_;
    std::vector<EdgeId> currentFacesBd_, nextFacesBd_;

    void firstStep_();
    void nextStep_();
};

ContourLeftFiller::ContourLeftFiller( const MeshTopology & topology ) : topology_( topology )
{
    filledFaces_.resize( topology.lastValidFace() + 1 );
}

void ContourLeftFiller::addContour( const std::vector<EdgeId> & contour )
{
    for ( auto e : contour )
        currentFacesBd_.push_back( e );
}

void ContourLeftFiller::firstStep_()
{
    MR_TIMER
        // first step is more complicated to ensure that we do not fill to the other side from the contour

        phmap::parallel_flat_hash_set<EdgeId> initialEdges;
    for ( EdgeId e : currentFacesBd_ )
        initialEdges.insert( e );

    std::vector<EdgeId> nextFacesBd;
    auto addNextStepEdge = [&]( EdgeId e )
    {
        if ( initialEdges.find( e ) != initialEdges.end() )
            return;
        nextFacesBd.push_back( e.sym() );
    };

    for ( EdgeId e : currentFacesBd_ )
    {
        //allow the contour to pass back and forth an edge
        if ( initialEdges.find( e.sym() ) != initialEdges.end() )
            continue;
        auto l = topology_.left( e );
        if ( !l.valid() )
            continue;
        if ( !filledFaces_.test_set( l ) )
        {
            addNextStepEdge( topology_.next( e ).sym() );
            addNextStepEdge( topology_.prev( e.sym() ) );
        }
    }

    currentFacesBd_ = std::move( nextFacesBd );
}

void ContourLeftFiller::nextStep_()
{
    nextFacesBd_.clear();
    for ( EdgeId e : currentFacesBd_ )
    {
        auto l = topology_.left( e );
        if ( !l.valid() )
            continue;
        if ( !filledFaces_.test_set( l ) )
        {
            nextFacesBd_.push_back( topology_.next( e ) );
            nextFacesBd_.push_back( topology_.prev( e.sym() ).sym() );
        }
    }

    currentFacesBd_.swap( nextFacesBd_ );
}

const FaceBitSet& ContourLeftFiller::fill()
{
    firstStep_();
    while ( !currentFacesBd_.empty() )
        nextStep_();
    return filledFaces_;
}

FaceBitSet fillContourLeft(const MeshTopology& topology, const EdgePath& contour)
{
    MR_TIMER
    ContourLeftFiller filler( topology );
    filler.addContour( contour );
    return filler.fill();
}

FaceBitSet fillContourLeft( const MeshTopology & topology, const std::vector<EdgePath> & contours )
{
    MR_TIMER
    ContourLeftFiller filler( topology );
    for ( auto & contour : contours )
        filler.addContour( contour );
    return filler.fill();
}

} // namespace MR
