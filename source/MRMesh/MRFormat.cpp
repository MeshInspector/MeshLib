#include "MRFormat.h"
#include "MRBitSetParallelFor.h"

std::string fmt::formatter<MR::BitSet>::toString( const MR::BitSet& bs )
{
    std::string result( bs.size(), '0' );
    MR::BitSetParallelFor( bs, [&] ( auto i )
    {
        result[i] = '1';
    } );
    return result;
}
