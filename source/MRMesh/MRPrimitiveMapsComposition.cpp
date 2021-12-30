#include "MRPrimitiveMapsComposition.h"
#include "MRVector.h"
#include "MRId.h"

namespace MR
{

template<typename Tag>
void mapsComposition( Vector<Id<Tag>, Id<Tag>>& a2b, const Vector<Id<Tag>, Id<Tag>>& b2c )
{
    for ( int i = 0; i < a2b.size(); ++i )
    {
        auto& bId = a2b[Id<Tag>( i )];
        if ( bId )
            bId = b2c[bId];
    }
}

template<typename Tag>
Vector<Id<Tag>, Id<Tag>> mapsComposition( const Vector<Id<Tag>, Id<Tag>>& a2b, const Vector<Id<Tag>, Id<Tag>>& b2c )
{
    auto a2c = a2b;
    mapsComposition( a2c, b2c );
    return a2c;
}

void vertMapsComposition( VertMap& a2b, const VertMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

VertMap vertMapsComposition( const VertMap& a2b, const VertMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

void edgeMapsComposition( EdgeMap& a2b, const EdgeMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

EdgeMap edgeMapsComposition( const EdgeMap& a2b, const EdgeMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

void faceMapsComposition( FaceMap& a2b, const FaceMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

FaceMap faceMapsComposition( const FaceMap& a2b, const FaceMap& b2c )
{
    return mapsComposition( a2b, b2c );
}

}