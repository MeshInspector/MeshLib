#include "MRPreciseSegmentIntersectionOrder3.h"
#include "MRHighPrecision.h"
#include <array>

namespace MR
{

namespace
{

/// computes signed volume of the parallelepiped ABC0
Int128 orient3dVolume( const Vector3i& a, const Vector3i& b, const Vector3i& c )
{
    return dot( Vector3i128{ a }, Vector3i128{ cross( Vector3i64{ b }, Vector3i64{ c } ) } );
}

/// computes signed volume of the parallelepiped ABCD
inline Int128 orient3dVolume( const Vector3i& a, const Vector3i& b, const Vector3i& c, const Vector3i& d )
    { return orient3dVolume( a - d, b - d, c - d ); }

[[maybe_unused]] bool allUnique( std::array<VertId, 8> a )
{
    std::sort( begin( a ), end( a ) );
    return std::adjacent_find( begin( a ), end( a ) ) == end( a );
}

} // anonymous namespace

bool segmentIntersectionOrder(
    const PreciseVertCoords segm[2],
    const PreciseVertCoords ta[3],
    const PreciseVertCoords tb[3] )
{
    assert( doTriangleSegmentIntersect( { ta[0], ta[1], ta[2], segm[0], segm[1] } ) );
    assert( doTriangleSegmentIntersect( { tb[0], tb[1], tb[2], segm[0], segm[1] } ) );
    assert( allUnique( { segm[0].id, segm[1].id, ta[0].id, ta[1].id, ta[2].id, tb[0].id, tb[1].id, tb[2].id } ) );

    // res = ( orient3d(ta,segm[0])*orient3d(tb,segm[1])   -   orient3d(tb,segm[0])*orient3d(ta,segm[1]) ) /
    //       ( orient3d(ta,segm[0])-orient3d(ta,segm[1]) ) * ( orient3d(tb,segm[0])-orient3d(tb,segm[1]) )
    const auto volTaOrg  = orient3dVolume( ta[0].pt, ta[1].pt, ta[2].pt, segm[0].pt );
    const auto volTaDest = orient3dVolume( ta[0].pt, ta[1].pt, ta[2].pt, segm[1].pt );
    const auto volTbOrg  = orient3dVolume( tb[0].pt, tb[1].pt, tb[2].pt, segm[0].pt );
    const auto volTbDest = orient3dVolume( tb[0].pt, tb[1].pt, tb[2].pt, segm[1].pt );

    const auto den0 = volTaOrg - volTaDest;
    const auto den1 = volTbOrg - volTbDest;
    bool changeResSign = false;
    if ( den0 != 0 && den1 != 0 )
        changeResSign = ( den0 < 0 ) != ( den1 < 0 );
    else
        assert( !"not implemented" );
    
    const auto nom = Int256( volTaOrg ) * Int256( volTbDest ) - Int256( volTbOrg ) * Int256( volTaDest );
    if ( nom != 0 )
        return ( nom > 0 ) != changeResSign;

    assert( !"not implemented" );
    return false;
}

} //namespace MR
