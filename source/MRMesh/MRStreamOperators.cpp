#include "MRStreamOperators.h"
#include "MRBitSet.h"
#include <cassert>

namespace MR
{

std::ostream& operator << ( std::ostream& s, const BitSet & bs )
{
    auto i = bs.size();
    while ( i > 0 )
    {
        --i;
        s.put( bs.test( i ) ? '1' : '0' );
    }
    return s;
}

std::istream& operator >> ( std::istream& s, BitSet & bs )
{
    bs.clear();
    for (;;)
    { 
        auto c = s.peek();
        if ( c != '0' && c != '1' )
            break;
        (void)s.get();
        bs.push_back( c == '1' );
    }
    bs.reverse();
    return s;
}

} //namespace MR
