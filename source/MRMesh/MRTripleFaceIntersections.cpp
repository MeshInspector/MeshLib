#include "MRTripleFaceIntersections.h"
#include "MRFaceFace.h"
#include "MRMeshTopology.h"
#include "MRTimer.h"
#include <MRPch/MRTBB.h>
#include <parallel_hashmap/phmap.h>

namespace MR
{

namespace
{

struct FaceFaceHash
{
    size_t operator()( const FaceFace& ff ) const
    {
        return 17 * size_t( ff.aFace ) + 23 * size_t( ff.bFace );
    }
};

} //anonymous namespace

std::vector<FaceFaceFace> findTripleFaceIntersections( const MeshTopology& topology, const ContinuousContours& selfContours )
{
    MR_TIMER;

    assert( ( selfContours.size() % 2 ) == 0 );
    std::vector<FaceFaceFace> res;
    if ( selfContours.empty() )
        return res;

    // both in HashSet and in vector: aFace.id < bFace.id
    HashSet<FaceFace, FaceFaceHash> ffHashSet;
    std::vector<FaceFace> ffVec;
    auto add = [&]( FaceFace ff )
    {
        assert( ff.aFace != ff.bFace );
        if ( ff.bFace < ff.aFace )
            std::swap( ff.aFace, ff.bFace );
        if ( ffHashSet.insert( ff ).second )
            ffVec.push_back( ff );
    };

    for ( int i = 0; i < selfContours.size(); i += 2 )
    {
        assert( selfContours[i].size() == selfContours[i+1].size() );
        const auto & contour = selfContours[i];
        assert( !contour.empty() );
        if ( auto r = topology.right( contour.front().edge ) )
            add( { r, contour.front().tri() } );
        for ( const auto& vet : contour )
            if ( auto l = topology.left( vet.edge ) )
                add( { l, vet.tri() } );
    }

    tbb::parallel_sort( begin( ffVec ), end( ffVec ) );

    for ( int i = 0; i < ffVec.size(); )
    {
        const auto fa = ffVec[i].aFace;
        for ( int j = i + 1; j < ffVec.size() && ffVec[j].aFace == fa; ++j )
        {
            const auto fb = ffVec[j].bFace;
            assert( fa < fb );
            assert( ffHashSet.count( { fa, fb } ) );
            for ( int k = j + 1; k < ffVec.size() && ffVec[k].aFace == fa; ++k )
            {
                const auto fc = ffVec[k].bFace;
                assert( fb < fc );
                assert( ffHashSet.count( { fa, fc } ) );
                if ( ffHashSet.count( { fb, fc } ) )
                    res.emplace_back( fa, fb, fc );
            }
        }

        for ( ++i; i < ffVec.size() && ffVec[i].aFace == fa; ++i )
            {}
    }

    return res;
}

} //namespace MR
