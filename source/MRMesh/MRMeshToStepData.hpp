#pragma once

#include "MRMeshToStep.h"
#include "MRMeshToStepSurf.hpp"
#include "MRMesh.h"
#include "MRProgressCallback.h"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <vector>

namespace MR
{

namespace MeshToStep
{

/// a part of the mesh lying on one surface
struct Region
{
    Surf surf;
    int seg = 0;      ///< the segment of the mesh given by the user
    int numFaces = 0;
    double area = 0;
};

/// the mesh in the form convenient for the algorithms below, and its subdivision on regions
struct Data
{
    const Mesh* mesh = nullptr;
    int nf = 0;
    std::vector<V3> pts;       ///< by vertex
    std::vector<Tri> tris;     ///< by face; {-1,-1,-1} for invalid faces
    std::vector<Tri> nbr;      ///< faces across the edges (v0v1, v1v2, v2v0), -1 if no face there
    std::vector<V3> normals, centers;
    std::vector<double> areas;
    std::vector<int> seg;      ///< user segment by face, -1 for invalid faces
    std::vector<int> sub;      ///< region by face
    std::vector<Region> regions;
    double tol = 0, total = 0, diag = 0;
    int numSegs = 0;

    bool valid( int f ) const { return tris[f][0] >= 0; }

    /// the neighbour across i-th edge if it is in the same segment
    int nbrSameSeg( int f, int i ) const
    {
        const int g = nbr[f][i];
        return g >= 0 && seg[g] == seg[f] ? g : -1;
    }

    void updateStats()
    {
        for ( auto& r : regions )
        {
            r.numFaces = 0;
            r.area = 0;
        }
        for ( int f = 0; f < nf; ++f )
            if ( valid( f ) )
            {
                ++regions[sub[f]].numFaces;
                regions[sub[f]].area += areas[f];
            }
    }

    std::vector<std::vector<int>> facesOfRegions() const
    {
        std::vector<std::vector<int>> res( regions.size() );
        for ( int f = 0; f < nf; ++f )
            if ( valid( f ) )
                res[sub[f]].push_back( f );
        return res;
    }

    /// all triangles' vertices of given faces, with repetitions
    std::vector<V3> triPoints( const std::vector<int>& faces ) const
    {
        std::vector<V3> res;
        res.reserve( faces.size() * 3 );
        for ( int f : faces )
            for ( int v : tris[f] )
                res.push_back( pts[v] );
        return res;
    }

    /// distinct vertices of given faces
    std::vector<int> vertsOf( const std::vector<int>& faces ) const
    {
        std::vector<int> res;
        res.reserve( faces.size() * 3 );
        for ( int f : faces )
            for ( int v : tris[f] )
                res.push_back( v );
        std::sort( res.begin(), res.end() );
        res.erase( std::unique( res.begin(), res.end() ), res.end() );
        return res;
    }

    /// positions of distinct vertices of given faces, evenly subsampled if there are more than maxNum of them
    std::vector<V3> pointsOf( const std::vector<int>& faces, size_t maxNum = SIZE_MAX ) const
    {
        const auto vs = vertsOf( faces );
        std::vector<V3> res;
        if ( vs.size() <= maxNum )
        {
            res.reserve( vs.size() );
            for ( int v : vs )
                res.push_back( pts[v] );
        }
        else
        {
            res.reserve( maxNum );
            for ( size_t i = 0; i < maxNum; ++i )
                res.push_back( pts[vs[i * ( vs.size() - 1 ) / ( maxNum - 1 )]] );
        }
        return res;
    }

    /// area-weighted average distance from the centers of given faces to the surface
    double centersDeviation( const Surf& s, const std::vector<int>& faces ) const
    {
        double sum = 0, sumA = 0;
        for ( int f : faces )
        {
            sum += std::abs( s.dist( centers[f] ) ) * areas[f];
            sumA += areas[f];
        }
        return sumA > 0 ? sum / sumA : 0;
    }

    /// facet normals must not deviate much from the normals of the surface
    bool normalsAgree( const Surf& s, int f, bool checkVerts ) const
    {
        if ( std::abs( dot( s.normal( centers[f] ), normals[f] ) ) <= cCos40 )
            return false;
        if ( checkVerts && s.type != SurfType::Plane && s.type != SurfType::BSpline )
        {
            // the middles of facet's edges must not be too far from the surface relative to its radius,
            // otherwise it is rather a prism or a pyramid than a coarse cylinder or cone
            for ( int i = 0; i < 3; ++i )
            {
                const V3 m = ( pts[tris[f][i]] + pts[tris[f][( i + 1 ) % 3]] ) / 2.0;
                double rad = s.r;
                if ( s.type == SurfType::Cone )
                {
                    const V3 dm = m - s.c;
                    rad = ( dm - s.a * dot( dm, s.a ) ).length();
                }
                else if ( s.type == SurfType::Torus )
                    rad = s.R + s.r;
                if ( std::abs( s.dist( m ) ) > 0.1 * rad )
                    return false;
            }
        }
        if ( checkVerts && s.type != SurfType::Plane && s.type != SurfType::BSpline )
            for ( int v : tris[f] )
            {
                if ( s.type == SurfType::Cone && ( pts[v] - s.c ).length() < 5 * tol )
                    continue; // no normal in the apex
                // the limit is below 30 degrees to distinguish a hexagonal prism from a cylinder
                if ( std::abs( dot( s.normal( pts[v] ), normals[f] ) ) < cCosVert )
                    return false;
            }
        return true;
    }

    bool faceOnSurf( const Surf& s, int f, double maxDist ) const
    {
        for ( int v : tris[f] )
            if ( !( std::abs( s.dist( pts[v] ) ) < maxDist ) )
                return false;
        return normalsAgree( s, f, true );
    }
};

/// prepares the data for the conversion of the mesh
Data makeData( const Mesh& mesh, const MeshToStepSettings& settings );

/// finds regions of the mesh lying on planes, cylinders, cones, spheres and tori by growing them from seed triangles;
/// \return false if the operation was canceled
bool findAnalyticRegions( Data& d, const ProgressCallback& cb );

/// improves the regions found by findAnalyticRegions, and replaces groups of unreliable regions with B-spline surfaces
void cleanRegions( Data& d );

/// creates boundary representation of the regions, and writes it in STEP format
MeshToStepStats writeBRep( const Data& d, const std::string& name, std::ostream& out );

} //namespace MeshToStep

} //namespace MR
