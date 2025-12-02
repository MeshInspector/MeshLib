#include "MRFormat.h"

#include "MRBitSet.h"

namespace MR
{

std::string format_as( const BitSet& bs )
{
    std::string result( '0', bs.size() );
    for ( auto i = bs.find_first(); i != BitSet::npos; i = bs.find_next( i ) )
        result[bs.size() - 1 - i] = '1';
    return result;
}

} // namespace MR
