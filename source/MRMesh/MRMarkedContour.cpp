#include "MRMarkedContour.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

namespace
{

bool firstLastMarked( const MarkedContour3f & in )
{
    if ( !in.marks.test( 0 ) )
        return false;

    if ( in.marks.find_last() + 1 != in.contour.size() )
        return false;

    return true;
}

} // anonymous namespace

MarkedContour3f resampled( const MarkedContour3f & in, float maxStep )
{
    MR_TIMER
    MarkedContour3f res;
    if ( in.contour.empty() )
        return res;

    assert( firstLastMarked( in ) );
    res.marks.autoResizeSet( res.contour.size() );
    res.contour.push_back( in.contour.front() );

    for ( size_t i = 0; i + 1 < in.contour.size(); )
    {
        auto i1 = i; //< will be next mark
        float distance = 0; //< distance to next mark
        for ( ++i1; i1 < in.contour.size(); ++i1 )
        {
            distance += ( in.contour[i1 - 1] - in.contour[i1] ).length();
            if ( in.marks.test( i1 ) )
                break;
        }
        assert( in.marks.test( i1 ) );

        const int numMidPoints = int( distance / maxStep );
        if ( numMidPoints > 0 )
        {
            const float step = distance / ( numMidPoints + 1 );
            float remDistance = step; //< till next sample point
            auto i2 = i;
            auto p = in.contour[i2];
            while ( i2 < i1 )
            {
                auto segmLen = ( in.contour[i2 + 1] - p ).length();
                if ( segmLen <= remDistance )
                {
                    remDistance -= segmLen;
                    p = in.contour[++i2];
                    continue;
                }
                const float a = remDistance / segmLen;
                p = ( 1 - a ) * p + a * in.contour[i2 + 1];
                res.contour.push_back( p );
                remDistance = step;
            }
        }
        i = i1;
        res.marks.autoResizeSet( res.contour.size() );
        res.contour.push_back( in.contour[i] );
    }

    assert( firstLastMarked( res ) );
    assert( in.marks.count() == res.marks.count() );
    return res;
}

TEST(MRMesh, MarkedContour)
{
    auto mc = markedContour( Contour3f{ Vector3f{ 0, 0, 0 }, Vector3f{ 1, 0, 0 } } );
    auto rc = resampled( mc, 2 );
    EXPECT_EQ( mc.contour, rc.contour );
    EXPECT_EQ( mc.marks, rc.marks );

    rc = resampled( mc, 0.4f );
    EXPECT_EQ( rc.contour.size(), 4 );
}

} //namespace MR
