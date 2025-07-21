#include "MRSolidOfRevolution.h"

#include "MRMesh.h"
#include "MRParallelFor.h"

namespace MR
{

Mesh makeSolidOfRevolution( const Contour2f& profile, int resolution )
{
    const auto isClosed = ( profile.front() == profile.back() );
    const auto profileSize = profile.size() - int( isClosed );

    std::vector<Vector3f> points( profileSize * resolution );
    ParallelFor( points, [&] ( size_t index )
    {
        const auto angle = 2.f * PI_F * (float)( index / profileSize ) / (float)resolution;
        const auto x = std::cos( angle ), y = std::sin( angle );

        const auto& pp = profile[index % profileSize];
        points[index] = {
            pp.x * x,
            pp.x * y,
            pp.y,
        };
    } );

    Triangulation t;
    t.reserve( 2 * ( profileSize - 1 ) * resolution );
    for ( auto y0 = 0; y0 < resolution; ++y0 )
    {
        const auto y1 = ( y0 + 1 ) % resolution;
        for ( auto x0 = 0; x0 + int( !isClosed ) < profileSize; ++x0 )
        {
            const auto x1 = ( x0 + 1 ) % profileSize;

            // for profile points with x = 0 (point is lying on the axis) use the same vertex
            #define FIND_VERTEX( y_, x_ ) VertId { int( bool( profile[x_].x ) ) * y_ * profileSize + x_ }
            const auto
                v00 = FIND_VERTEX( y0, x0 ),
                v01 = FIND_VERTEX( y0, x1 ),
                v10 = FIND_VERTEX( y1, x0 ),
                v11 = FIND_VERTEX( y1, x1 );
            #undef FIND_VERTEX

            if ( points[v00] != points[v10] )
                t.push_back( { v00, v10, v11 } );
            if ( points[v01] != points[v11] )
                t.push_back( { v00, v11, v01 } );
        }
    }

    return Mesh::fromTriangles( std::move( points ), t );
}

} // namespace MR
