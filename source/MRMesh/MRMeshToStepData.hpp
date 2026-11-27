#pragma once

#include "MRMeshToStep.h"
#include "MRMeshToStepSurf.hpp"
#include "MRMesh.h"
#include "MRProgressCallback.h"

#include <algorithm>
#include <ostream>
#include <string>

namespace MR
{

namespace MeshToStep
{

/// a part of the mesh bounded by the given edges, becoming one face of the result
struct Region
{
    Surf surf;
    double rms = 0;     ///< root-mean-square distance from the vertices of the region to its surface
    double maxDist = 0; ///< maximum distance from the vertices of the region to its surface
    int facetOf = -1;   ///< if not negative, the region is this one triangle, and its plane passes through the vertices wherever they are moved
};

/// the mesh in the form convenient for the algorithms below, and its subdivision on regions
struct Data
{
    const Mesh* mesh = nullptr;
    int nf = 0;
    Vector<V3> pts;       ///< by vertex; the vertices on the boundaries of regions are moved on the surfaces of all regions around them
    Vector<Tri> tris;     ///< by face; {-1,-1,-1} for invalid faces
    Vector<Tri> nbr;      ///< faces across the edges (v0v1, v1v2, v2v0), -1 if no face there
    Vector<V3> normals, centers;
    Vector<double> areas;
    Vector<int> sub;      ///< region by face, -1 for invalid faces
    Vector<Region> regions;
    double tol = 0;       ///< root-mean-square distance from the vertices to an analytic surface, above which a B-spline surface is tried
    Vector<double> snapErr; ///< by vertex: the maximum distance from a vertex on the boundaries of regions to their surfaces
    double edgeTol = 0;   ///< the precision of the curves and the vertices of the result, where all surfaces meet exactly
    double diag = 0;

    bool valid( int f ) const { return tris[f][0] >= 0; }

    Vector<Vector<int>> facesOfRegions() const
    {
        Vector<Vector<int>> res( regions.size() );
        for ( int f = 0; f < nf; ++f )
            if ( valid( f ) )
                res[sub[f]].push_back( f );
        return res;
    }

    /// distinct vertices of given faces
    Vector<int> vertsOf( const Vector<int>& faces ) const
    {
        Vector<int> res;
        res.reserve( faces.size() * 3 );
        for ( int f : faces )
            for ( int v : tris[f] )
                res.push_back( v );
        std::sort( res.begin(), res.end() );
        res.erase( std::unique( res.begin(), res.end() ), res.end() );
        return res;
    }
};

/// prepares the data for the conversion of the mesh: every connected part of triangles not separated by faceBoundaries is a region
Data makeData( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, const MeshToStepSettings& settings );

/// finds the surface of every region;
/// \return false if the operation was canceled
bool fitRegions( Data& d, const ProgressCallback& cb );

/// moves the vertices on the boundaries of regions to the closest points lying on the surfaces of all regions around them
void snapBoundaries( Data& d );

/// creates boundary representation of the regions, and writes it in STEP format
MeshToStepStats writeBRep( const Data& d, const std::string& name, std::ostream& out );

} //namespace MeshToStep

} //namespace MR
