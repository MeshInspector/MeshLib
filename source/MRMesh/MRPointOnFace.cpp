#include "MRPointOnFace.h"

#include <iostream>

namespace MR
{

std::ostream& operator<<( std::ostream& s, const PointOnFace& pof )
{
    return s << pof.face << '\n' << pof.point;
}

std::istream& operator>>( std::istream& s, PointOnFace& pof )
{
    int a;
    s >> a >> pof.point;
    pof.face = FaceId( a );
    return s;
}

}
