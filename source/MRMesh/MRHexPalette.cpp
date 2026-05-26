#include "MRHexPalette.h"
#include "MRVector3.h"

#include <cassert>

namespace MR
{

HexPalette::HexPalette()
{
    static constexpr int CORNER_COLORS = 6;
    static constexpr int SIDE_COLORS = 5; // num colors between two corner colors + 1
    // for any color c: dot( c, [1,1,1] ) = 1
    static const Vector3f cornerColors[CORNER_COLORS + 1] =
    {
        { 1.0, 0.0, 0.0 },
        { 0.5, 0.5, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.5, 0.5 },
        { 0.0, 0.0, 1.0 },
        { 0.5, 0.0, 0.5 },
        { 1.0, 0.0, 0.0 }
    };
    colors.reserve( CORNER_COLORS * SIDE_COLORS );
    for ( int corner = 0; corner < CORNER_COLORS; ++corner )
    {
        for ( int i = 0; i < SIDE_COLORS; ++i )
        {
            auto v = lerp( cornerColors[corner], cornerColors[corner+1], float(i) / SIDE_COLORS );
            colors.emplace_back( v );
        }
    }
    assert( colors.size() == CORNER_COLORS * SIDE_COLORS );
}

} // namespace MR
