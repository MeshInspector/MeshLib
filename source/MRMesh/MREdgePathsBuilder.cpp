#include "MREdgePathsBuilder.h"
#include "MRTimer.h"

namespace MR
{

EdgePath BuilderOfSmallestMetricPathBiDir::run(
    const TerminalVertex * starts, int numStarts,
    const TerminalVertex * finishes, int numFinishes,
    VertId * outPathStart, VertId * outPathFinish, float maxPathMetric )
{
    MR_TIMER
    assert( numStarts > 0 && numFinishes > 0 );

    VertId join;
    float joinPathMetric = maxPathMetric;

    bs_.clear();
    for ( int si = 0; si < numStarts; ++si )
        bs_.addStart( starts[si].v, starts[si].metric );

    bf_.clear();
    for ( int fi = 0; fi < numFinishes; ++fi )
        bf_.addStart( finishes[fi].v, finishes[fi].metric );

    bool keepGrowing = true;
    for (;;)
    {
        auto ds = bs_.doneDistance();
        auto df = bf_.doneDistance();
        if ( keepGrowing && join && joinPathMetric <= ds + df )
        {
            keepGrowing = false;
        }
        if ( ds <= df )
        {
            if ( ds >= FLT_MAX )
                break;
            auto c = bs_.reachNext();
            if ( !c.v )
                continue;
            if ( keepGrowing )
                bs_.addOrgRingSteps( c );
            if ( auto info = bf_.getVertInfo( c.v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = c.v;
                    joinPathMetric = newMetric;
                }
            }
        }
        else
        {
            auto c = bf_.reachNext();
            if ( !c.v )
                continue;
            if ( keepGrowing )
                bf_.addOrgRingSteps( c );
            if ( auto info = bs_.getVertInfo( c.v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = c.v;
                    joinPathMetric = newMetric;
                }
            }
        }
    }

    EdgePath res;
    const auto & topology = bs_.topology();
    if ( join )
    {
        res = bs_.getPathBack( join );
        reverse( res );
        auto tail = bf_.getPathBack( join );
        res.insert( res.end(), tail.begin(), tail.end() );
        assert( isEdgePath( topology, res ) );

        if ( res.empty() )
        {
            if ( outPathStart )
                *outPathStart = join;
            if ( outPathFinish )
                *outPathFinish = join;
        }
        else
        {
            assert( numStarts > 1 || topology.org( res.front() ) == starts[0].v );
            assert( numFinishes > 1 || topology.dest( res.back() ) == finishes[0].v );

            if ( outPathStart )
                *outPathStart = topology.org( res.front() );
            if ( outPathFinish )
                *outPathFinish = topology.dest( res.back() );
        }
    }
    return res;
}

EdgePath BuilderOfSmallestMetricPathBiDir::run(
    VertId start, VertId finish,
    VertId * outPathStart, VertId * outPathFinish, float maxPathMetric )
{
    TerminalVertex s{ start, 0 };
    TerminalVertex f{ finish, 0 };
    return run( &s, 1, &f, 1, outPathStart, outPathFinish, maxPathMetric );
}

} // namespace MR
