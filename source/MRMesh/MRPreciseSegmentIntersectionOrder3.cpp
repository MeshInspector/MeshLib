#include "MRPreciseSegmentIntersectionOrder3.h"

namespace MR
{

bool segmentIntersectionOrder(
    const PreciseVertCoords segm[2],
    const PreciseVertCoords ta[3],
    const PreciseVertCoords tb[3] )
{
    // res = ( mixed(tb,org)*mixed(ta,dest)   -   mixed(ta,org)*mixed(tb,dest) ) /
    //       ( mixed(ta,org)-mixed(ta,dest) ) * ( mixed(tb,org)-mixed(tb,dest) )
}

} //namespace MR
