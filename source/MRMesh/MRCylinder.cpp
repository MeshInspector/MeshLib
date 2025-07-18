#include "MRCylinder.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"

namespace MR
{

Mesh makeCylinder( float radius, float length, int resolution )
{
    std::vector<Vector3f> points(2 * resolution + 2);
    auto step = 2.0f * PI_F / float(resolution);

    for (int h = 0; h < 2; ++h)
    {
        points[h] = Vector3f(0.0f, 0.0f, float(h)*length);
        for (int anglei = 0; anglei < resolution; ++anglei)
        {
            float angle = anglei * step;
            auto ind = anglei + resolution * h + 2;
            points[ind].x = radius * cosf(angle);
            points[ind].y = radius * sinf(angle);
            points[ind].z = h * length;
        }
    }

    Triangulation t;
    t.reserve( 4 * resolution );
    for ( int i = 0; i < resolution; ++i )
    {
        t.push_back( { 0_v, VertId(((i + 1) % resolution) + 2), VertId(i + 2) } );
        t.push_back( { 1_v, VertId(i + 2 + resolution), VertId(((i + 1) % resolution) + 2 + resolution) } );
        t.push_back( { VertId(i + 2), VertId(((i + 1) % resolution) + 2), VertId(i + 2 + resolution) } );
        t.push_back( { VertId(((i + 1) % resolution) + 2), VertId(((i + 1) % resolution) + 2 + resolution),
            VertId(i + 2 + resolution) } );
    }

    return Mesh::fromTriangles( std::move( points ), t );
}

Mesh makeOpenCylinder( float radius, float z1, float z2, int numCircleSegments )
{
    std::vector<Vector3f> points( 2 * numCircleSegments );

    for ( bool side : { false, true } )
    {
        float z = side ? z2 : z1;

        for ( int i = 0; i < numCircleSegments; ++i )
        {
            float angle = i * 2 * PI_F / numCircleSegments;
            int index = i + numCircleSegments * side;
            points[index].x = radius * std::cos( angle );
            points[index].y = radius * std::sin( angle );
            points[index].z = z;
        }
    }

    Triangulation t;
    t.reserve( 2 * numCircleSegments );
    for ( int i = 0; i < numCircleSegments; ++i )
    {
        VertId a( i );
        VertId b( ( i + 1 ) % numCircleSegments );
        VertId c( i + numCircleSegments );
        VertId d( ( i + 1 ) % numCircleSegments + numCircleSegments );

        t.push_back( { a, b, c } );
        t.push_back( { b, d, c } );
    }

    return Mesh::fromTriangles( std::move( points ), t );
}

Mesh makeOpenCone( float radius, float zApex, float zBase, int numCircleSegments )
{
    std::vector<Vector3f> points( numCircleSegments + 1 );

    for ( int i = 0; i < numCircleSegments; ++i )
    {
        float angle = i * 2 * PI_F / numCircleSegments;
        points[i].x = radius * std::cos( angle );
        points[i].y = radius * std::sin( angle );
        points[i].z = zBase;
    }
    points[numCircleSegments] = Vector3f( 0, 0, zApex );

    Triangulation t;
    t.reserve( 2 * numCircleSegments );
    for ( int i = 0; i < numCircleSegments; ++i )
    {
        VertId a( i );
        VertId b( ( i + 1 ) % numCircleSegments );
        VertId c( numCircleSegments );
        if ( zBase > zApex )
            std::swap( a, b );
        t.push_back( { a, b, c } );
    }

    return Mesh::fromTriangles( std::move( points ), t );
}

Mesh makeCylinderAdvanced( float radius0, float radius1, float start_angle,
                               float arc_size, float length, int resolution )
{
    int cap0 = 0, cap1 = 0;
    bool slice = false;

    if (radius0 != 0)
    {
        cap0 = resolution;
    }

    if (radius1 != 0)
    {
        cap1 = resolution;
    }

    if (arc_size >= 2.0f * PI_F)
        arc_size = 2.0f * PI_F;
    else if (arc_size <= -2.0f * PI_F)
        arc_size = -2.0f * PI_F;
    else
    {
        if (cap0) ++cap0;
        if (cap1) ++cap1;
        slice = true;
    }

    std::vector<Vector3f> points(cap0 + cap1 + 2);
    auto step = arc_size / float(resolution);
    int ver_id = 0;

    points[ver_id++] = Vector3f(0.0f, 0.0f, 0.0f);
    points[ver_id++] = Vector3f(0.0f, 0.0f, length);

    for (int anglei = 0; anglei < cap0; ++anglei)
    {
        float angle = start_angle + anglei * step;
        points[ver_id].x = radius0 * cosf(angle);
        points[ver_id].y = radius0 * sinf(angle);
        points[ver_id].z = 0;
        ++ver_id;
    }

    for (int anglei = 0; anglei < cap1; ++anglei)
    {
        float angle = start_angle + anglei * step;
        points[ver_id].x = radius1 * cosf(angle);
        points[ver_id].y = radius1 * sinf(angle);
        points[ver_id].z = length;
        ++ver_id;
    }

    Triangulation t;
    t.reserve(2 * (cap0 + cap1) + 4 * slice);

    for (int i = 0; i < resolution; ++i)
    {
        if (cap0)
        {
            t.push_back( { VertId(0), VertId(((i + 1) % cap0) + 2), VertId(i + 2) } );
            t.push_back( { VertId(i + 2), VertId(((i + 1) % cap0) + 2), VertId(cap1 ? i + 2 + cap0 : 1) } );
        }
        if (cap1)
        {
            t.push_back( { VertId(1), VertId(i + 2 + cap0), VertId(((i + 1) % cap1) + 2 + cap0) } );
            if (cap0)
            {
                t.push_back( { VertId(((i + 1) % cap1) + 2), VertId(((i + 1) % cap1) + 2 + cap0), VertId(i + 2 + cap0) } );
            }
            else
            {
                t.push_back( { VertId(0), VertId(((i + 1) % cap1) + 2), VertId(i + 2 + cap0) } );
            }
        }
    }
    if (slice)
    {
        if (cap0)
        {
            t.push_back( { VertId(0), VertId(2), VertId(cap1 ? 2 + cap0 : 1) } );
            t.push_back( { VertId(resolution + 2), VertId(0), VertId(cap1 ? resolution + 2 + cap0 : 1) } );
        }
        if (cap1)
        {
            t.push_back( { VertId(1), VertId(0), VertId(2 + cap0) } );
            t.push_back( { VertId(0), VertId(1), VertId(resolution + 2 + cap0) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

Mesh makeCone( float radius0, float length, int resolution )
{
    return makeCylinderAdvanced( radius0, 0.f, 0.f, 2.0f * PI_F, length, resolution );
}

Mesh makeSolidOfRevolution( const Contour2f& profile, int resolution )
{
    std::vector<Vector3f> points;
    points.reserve( profile.size() * resolution );
    for ( auto i = 0; i < resolution; ++i )
    {
        const auto angle = 2.f * PI_F * (float)i / (float)resolution;
        const auto x = std::cos( angle ), y = std::sin( angle );
        for ( const auto& p : profile )
        {
            points.emplace_back(
                p.x * x,
                p.x * y,
                p.y
            );
        }
    }

    Triangulation t;
    t.reserve( 2 * ( profile.size() - 1 ) * resolution );
    for ( auto y0 = 0; y0 < resolution; ++y0 )
    {
        const auto y1 = ( y0 + 1 ) % resolution;
        for ( auto x0 = 0; x0 + 1 < profile.size(); ++x0 )
        {
            const auto x1 = x0 + 1;

            // for profile points with x = 0 (point is lying on the axis) use the same vertex
            #define FIND_VERTEX( y_, x_ ) VertId { int( bool( profile[x_].x ) ) * y_ * profile.size() + x_ }
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

}
