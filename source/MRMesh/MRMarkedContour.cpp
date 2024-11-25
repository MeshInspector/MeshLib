#include "MRMarkedContour.h"
#include "MRTimer.h"

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

MarkedContour3f resampled( const MarkedContour3f & in, float /*maxStep*/ )
{
    MR_TIMER
    assert( firstLastMarked( in ) );
    MarkedContour3f res;

    assert( firstLastMarked( res ) );
    assert( in.marks.count() == res.marks.count() );
    return res;
}

} //namespace MR
