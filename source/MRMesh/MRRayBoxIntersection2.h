#pragma once
#include "MRBox.h"
#include "MRIntersectionPrecomputes2.h"

namespace MR
{

template<typename T>
bool rayBoxIntersect( const Box2<T>& box, const Vector2<T> & rayOrigin, T & t0, T & t1, const IntersectionPrecomputes2<T>& prec )
{
    const Vector2i& sign = prec.sign;

    // compare and update x-dimension with t0-t1
    t1 = std::min( (box[sign.x].x - rayOrigin.x) * prec.invDir.x, t1 );
    t0 = std::max( (box[1 - sign.x].x - rayOrigin.x) * prec.invDir.x, t0 );

    // compare and update y-dimension with t0-t1
    t1 = std::min( (box[sign.y].y - rayOrigin.y) * prec.invDir.y, t1 );
    t0 = std::max( (box[1 - sign.y].y - rayOrigin.y) * prec.invDir.y, t0 );

    return t0 <= t1;
}

template <typename T = float>
bool rayBoxIntersect( const Box2<T>& box, const Line2<T>& line, T t0, T t1 )
{
    IntersectionPrecomputes2<T> prec( line.d );
    return rayBoxIntersect( box, line, t0, t1, prec );
}

/// \}

}
