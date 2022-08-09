#include "MRTorus.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"
#include <cmath>

namespace MR
{
MR::Mesh makeTorus( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = 2 * i * PI_F / secondaryResolution;
        for ( j = 0; j < primaryResolution; ++j )
        {
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::sin( b );
            points[k].z = secondaryRadius * std::sin( a );
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;
    int triangleCount = 2 * vertexCount;
    t.reserve( triangleCount );

    for ( i = 0; i < secondaryResolution; ++i )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + secondaryResolution - 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + primaryResolution - 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

MR::Mesh makeOuterHalfTorus( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = i * PI_F / secondaryResolution + PI_F / 2 ;
        for ( j = 0; j < primaryResolution; ++j )
        {
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::sin( b );
            points[k].z = secondaryRadius * std::sin( a );
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;

    int triangleCount = 2 * vertexCount - 2 * primaryResolution;
    t.reserve( triangleCount );

    for ( i = 0; i < secondaryResolution - 1; ++i )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + ( j + 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

MR::Mesh makeTorusWithUndercut( float primaryRadius, float secondaryRadiusInner, float secondaryRadiusOuter, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = 2.f * i * PI_F / secondaryResolution;
        auto sina = std::sin( a );
        auto secRadius = sina > 0.f ? secondaryRadiusOuter : secondaryRadiusInner;
        for ( j = 0; j < primaryResolution; ++j )
        {
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - secRadius * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - secRadius * std::cos( a ) ) * std::sin( b );
            points[k].z = ( sina > 0.f ? secRadius : -secRadius ) * sina;
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;

    int triangleCount = 2 * vertexCount;
    t.reserve( triangleCount );

    for ( i = 0; i < secondaryResolution; ++i )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + ( j + 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

MR::Mesh makeTorusWithSpikes( float primaryRadius, float secondaryRadiusInner, float secondaryRadiusOuter, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    const int spikeFreq = 23;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = 2.f * i * PI_F / secondaryResolution;
        for ( j = 0; j < primaryResolution; ++j )
        {
            bool isSpike = ( ( i * secondaryResolution + j ) % spikeFreq == 0 );
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - ( isSpike ? secondaryRadiusOuter : secondaryRadiusInner ) * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - ( isSpike ? secondaryRadiusOuter : secondaryRadiusInner ) * std::cos( a ) ) * std::sin( b );
            points[k].z = ( isSpike ? secondaryRadiusOuter : secondaryRadiusInner ) * std::sin( a );
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;

    int triangleCount = 2 * vertexCount - 2 * primaryResolution;
    t.reserve( triangleCount );

    for ( i = 0; i < secondaryResolution; ++i )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + ( j + 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

MR::Mesh makeTorusWithComponents( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = 2 * i * PI_F / secondaryResolution;
        for ( j = 0; j < primaryResolution; ++j )
        {
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::sin( b );
            points[k].z = secondaryRadius * std::sin( a );
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;

    int triangleCount = 2 * vertexCount - 2 * primaryResolution;
    t.reserve( triangleCount );

    for ( i = 0; i < secondaryResolution - 1; i+=2 )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + ( j + 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

MR::Mesh makeTorusWithSelfIntersections( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution,
                    std::vector<Vector3f>* pointsOut )
{
    int i, j, k;
    float a, b;

    int vertexCount = primaryResolution * secondaryResolution;
    std::vector<Vector3f> points( vertexCount );
    bool pSet = false;
    if ( pointsOut )
        pointsOut->resize( primaryResolution );
    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        a = 2 * i * PI_F / secondaryResolution;
        for ( j = 0; j < primaryResolution; ++j )
        {
            b = 2 * j * PI_F / primaryResolution;

            points[k].x = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::cos( b );
            points[k].y = ( primaryRadius - secondaryRadius * std::cos( a ) ) * std::sin( b );
            points[k].z = secondaryRadius * std::sin( 2.f * a );
            ++k;
            if ( pointsOut && !pSet )
                ( *pointsOut )[j] = {primaryRadius * std::cos( b ),primaryRadius * std::sin( b ),0.0f};
        }
        pSet = true;
    }

    Triangulation t;

    int triangleCount = 2 * vertexCount - 2 * primaryResolution;
    t.reserve( triangleCount );

    k = 0;
    for ( i = 0; i < secondaryResolution; ++i )
    {
        for ( j = 0; j < primaryResolution; ++j )
        {
            t.push_back( {
                VertId( i * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ) } );

            t.push_back( {
                VertId( i * primaryResolution + ( j + 1 ) % primaryResolution ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + j ),
                VertId( ( i + 1 ) % secondaryResolution * primaryResolution + ( j + 1 ) % primaryResolution ) } );
        }
    }

    return Mesh::fromTriangles( std::move(points), t );
}

}
